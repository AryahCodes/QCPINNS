import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from src.utils.logger import Logging
from src.nn.DVQuantumLayer import DVQuantumLayer


class DVPDESolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=None):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.num_qubits = self.args["num_qubits"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True
        self.classic_network = self.args["classic_network"]
        self.total_training_time = 0
        self.total_memory_peak = 0
        
        # ⚡ Get loss weights from args
        self.loss_pde_weight = self.args.get("loss_pde_weight", 1.0)
        self.loss_bc_weight = self.args.get("loss_bc_weight", 100.0)
        
        if self.encoding == "amplitude":
            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)
        else:
            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)

        if "ionq" in self.args.get("qml_device", "").lower():
            # IonQ: single measurement
            quantum_output_dim = 1
        else:
            # Local: multiple measurements
            quantum_output_dim = self.num_qubits
        self.postprocessor = nn.Sequential(
            nn.Linear(quantum_output_dim, self.classic_network[-2]).to(self.device),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(self.device),
        ).to(self.device)

        self.activation = nn.Tanh()

        self.num_qubits = args["num_qubits"]
        if "ionq" in self.args.get("qml_device", "").lower():
            from src.nn.QNeuralNetwork1 import QNeuralNetwork1
            self.quantum_layer = QNeuralNetwork1(self.args)
            self.logger.print("✅ Using QNeuralNetwork1 (IonQ backend)")
        else:
            from src.nn.DVQuantumLayer import DVQuantumLayer
            self.quantum_layer = DVQuantumLayer(self.args)
            self.logger.print("🧠 Using DVQuantumLayer (local simulator)")


        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args["lr"]
        )

        # ⚡ FIXED: Less aggressive scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500
        )

        self.loss_fn = torch.nn.MSELoss()

        self._initialize_logging()
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network
        Args:
            x: Spatial coordinates (batch_size, input_dim)
        Returns:
            PDE solution values (batch_size, output_dim)
        """

        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            
            # Classical preprocessing: (batch_size, input_dim) -> (batch_size, num_qubits)
            preprocessed = self.preprocessor(x)

            if self.draw_quantum_circuit_flag:
                self.draw_quantum_circuit(preprocessed)
                self.draw_quantum_circuit_flag = False

            # Quantum processing
            quantum_out = self.quantum_layer(preprocessed)
            
            # Convert list to tensor if needed
            if isinstance(quantum_out, list):
                quantum_out = torch.stack(quantum_out)
            
            quantum_out = quantum_out.to(dtype=torch.float32, device=self.device)

            # ⚡⚡⚡ CRITICAL FIX ⚡⚡⚡
            # quantum_out shape is [num_qubits, batch_size] from PennyLane
            # We need [batch_size, num_qubits] for the postprocessor
            # Simple transpose is the correct operation!
            if hasattr(self.quantum_layer, '__class__') and \
            self.quantum_layer.__class__.__name__ == 'QNeuralNetwork1':
                # QNeuralNetwork1 returns [batch_size, 1] - single measurement
                quantum_features = quantum_out
                expected_shape = (x.shape[0], 1)  # ← FIXED!
            else:
                # DVQuantumLayer returns [batch_size, num_qubits] - multiple measurements
                quantum_features = quantum_out
                expected_shape = (x.shape[0], self.num_qubits)

            if quantum_features.shape != expected_shape:
                self.logger.print(f"WARNING: quantum_features shape {quantum_features.shape} != expected {expected_shape}")
                # Force correct shape if needed
                quantum_features = quantum_features.reshape(expected_shape)

            # Classical postprocessing: (batch_size, num_qubits) -> (batch_size, output_dim)
            classical_out = self.postprocessor(quantum_features)
            
            return classical_out

        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            self.logger.print(f"Input shape: {x.shape}")
            if 'preprocessed' in locals():
                self.logger.print(f"Preprocessed shape: {preprocessed.shape}")
            if 'quantum_out' in locals():
                self.logger.print(f"Quantum out shape: {quantum_out.shape}")
            if 'quantum_features' in locals():
                self.logger.print(f"Quantum features shape: {quantum_features.shape}")
            raise

    def save_state(self, path=None):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "quantum_params": self.quantum_layer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }

        if path is None:
            model_path = os.path.join(self.log_path, "model.pth")
        else:    
            model_path = path

        with open(model_path, "wb") as f:
            torch.save(state, f)

        self.logger.print(f"Model state saved to {model_path}")

    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            state = torch.load(f, map_location=map_location)
        return state

    def draw_quantum_circuit(self, x):
        if self.draw_quantum_circuit_flag:
            try:
                self.logger.print("The circuit used in the study:")
                if hasattr(self.quantum_layer, "circuit"):
                    fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0])
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()
                    print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")