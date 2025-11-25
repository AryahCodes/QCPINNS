import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pennylane as qml
from src.utils.logger import Logging # <--- Added missing import


class CVPDESolver(nn.Module):
    """Hybrid Classical-CV Quantum Neural Network for PDE Solving"""

    def __init__(self, args, logger: Logging, data=None, device="cpu"): # <--- Fixed type hint
        super().__init__()

        self.logger = logger
        self.device = device
        self.data = data
        self.input_dim = args["input_dim"]
        self.num_qubits = args["num_qubits"]
        self.hidden_dim = args["hidden_dim"]
        self.output_dim = args["output_dim"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.epochs = args["epochs"]
        self.args = args
        self.batch_size = args["batch_size"]
        self.draw_quantum_circuit_flag = True # <--- Added this line
        self.log_path = self.logger.get_output_dir()
        self.model_path = os.path.join(self.log_path, "model.pth")
        
        try:
            class_name = self.args.get("class")
            
            # --- New Qubit-based (IonQ) classes ---
            if class_name == "QNeuralNetwork1":
                from src.nn.QNeuralNetwork1 import QNeuralNetwork1
                self.logger.print("Using QNeuralNetwork1 (IonQ)")
                self.quantum_layer = QNeuralNetwork1(self.args) # <--- CORRECT
            elif class_name == "QNeuralNetwork2":
                from src.nn.QNeuralNetwork2 import QNeuralNetwork2
                self.logger.print("Using QNeuralNetwork2 (IonQ)")
                self.quantum_layer = QNeuralNetwork2(self.args) # <--- FIXED
            elif class_name == "QNeuralNetwork3":
                from src.nn.QNeuralNetwork3 import QNeuralNetwork3
                self.logger.print("Using QNeuralNetwork3 (IonQ)")
                self.quantum_layer = QNeuralNetwork3(self.args) # <--- FIXED

            # --- Original CV (Strawberry Fields) classes ---
            elif class_name == "CVNeuralNetwork2":
                from src.nn.CVNeuralNetwork2 import CVNeuralNetwork2
                self.logger.print("Using CVNeuralNetwork2")
                self.quantum_layer = CVNeuralNetwork2(self.args) # <--- Standardized
            elif class_name == "GSRandomCVQNN2":
                from src.nn.CVNeuralNetwork3 import CVNeuralNetwork3
                self.logger.print("Using CVNeuralNetwork3")
                self.quantum_layer = CVNeuralNetwork3(self.args) # <--- Standardized
            else: # Default or "CVNeuralNetwork1"
                from src.nn.CVNeuralNetwork1 import CVNeuralNetwork1
                self.logger.print("Using CVNeuralNetwork1")
                self.quantum_layer = CVNeuralNetwork1(self.args) # <--- Standardized
        
        except Exception as e:
            self.logger.print(f"Failed to initialize quantum layer: {str(e)}")
            raise

        self.preprocessor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim).to(self.device),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_qubits).to(self.device),
        ).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.hidden_dim).to(self.device),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim).to(self.device),
        ).to(self.device)

        # ... (rest of your file is fine, I'm pasting it) ...
        
        if self.args.get("class", "CVNeuralNetwork") == "GSRandomCVQNN2":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["lr"]
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["lr"], weight_decay=0.001
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.9, patience=800, min_lr=1e-6
            )

        self.loss_fn = torch.nn.MSELoss()
        self.loss_history = []
        self.params = None # <--- This was in your file
        self._initialize_logging()
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.postprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _initialize_logging(self):
        if self.num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2")
        if self.num_quantum_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        self.log_path = self.logger.get_output_dir()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            preprocessed = self.preprocessor(x)
            quantum_out = self.quantum_layer(preprocessed) # .to(dtype=torch.float32, device=self.device)
            classical_out = self.postprocessor(quantum_out)
            return classical_out
        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    def save_state(self):
        state = {
            "args": self.args,
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "model_path": self.model_path,
        }
        with open(self.model_path, "wb") as f:
            torch.save(state, f)
            self.logger.print(f"Model state saved to {self.model_path}")

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
                # This check for 'params' is from your old file, but QNN1 doesn't have it.
                # Let's check for the qnode on the quantum_layer instead.
                if hasattr(self.quantum_layer, 'circuit'):
                    fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0]) # Draw with one sample
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()
                    self.draw_quantum_circuit_flag = False
                    self.logger.print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")