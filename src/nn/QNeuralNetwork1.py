import pennylane as qml
import torch
import torch.nn as nn
import pennylane_ionq
import os



class QNeuralNetwork1(nn.Module):
    """
    Qubit-based neural network for IonQ.
    Implements a parameterized quantum circuit with RX, RY, RZ rotations and CNOT entanglement.
    """

    def __init__(self, args: dict):
        super().__init__()
        
        self.n_wires = args["num_qubits"]
        self.num_layers = args["num_quantum_layers"]
        self.torch_device = args["device"]
        
        qml_device_name = args.get("qml_device", "default.qubit")
        
        # For IonQ with PennyLane
        if "ionq" in qml_device_name.lower():
            import pennylane_ionq
            self.dev = qml.device(
                qml_device_name,  # or "ionq.qpu" for real hardware
                wires=self.n_wires,
                shots=args.get("shots", 1000),
            )
            diff_method = "parameter-shift"
        else:
            # Local simulator (default.qubit)
            shots = args.get("shots")
            if shots is None:
                # No shots - use exact simulation
                self.dev = qml.device(qml_device_name, wires=self.n_wires)
            else:
                # With shots - use sampling
                self.dev = qml.device(qml_device_name, wires=self.n_wires, shots=shots)
            diff_method = "best"
        
        # Initialize parameters with small random values
        active_sd = 0.0001   # For rotations that affect amplitudes
        passive_sd = 0.1     # For phase rotations
        
        # Trainable parameters for each layer
        self.theta_rx = nn.Parameter(
            torch.randn(self.num_layers, self.n_wires, device=self.torch_device) * active_sd
        )
        self.theta_ry = nn.Parameter(
            torch.randn(self.num_layers, self.n_wires, device=self.torch_device) * passive_sd
        )
        self.theta_rz = nn.Parameter(
            torch.randn(self.num_layers, self.n_wires, device=self.torch_device) * passive_sd
        )

        # Create quantum circuit
        self.circuit = qml.QNode(
            self._quantum_circuit, 
            self.dev, 
            interface="torch", 
            diff_method=diff_method
        )
        
        self.activation = nn.Tanh()

    def _quantum_circuit(self, inputs):
        """
        Quantum circuit implementation.
        
        Args:
            inputs: Input features (num_qubits,) - one value per qubit
            
        Returns:
            List of expectation values for each qubit
        """
        # Data encoding: RX rotation with input data
        for w, x in enumerate(inputs):
            qml.RX(x, wires=w)
        
        # Parameterized layers
        for l in range(self.num_layers):
            # Single-qubit rotations
            for w in range(self.n_wires):
                qml.RX(self.theta_rx[l, w], wires=w)
                qml.RY(self.theta_ry[l, w], wires=w)
            
            # Entanglement layer (circular CNOT chain)
            for w in range(self.n_wires):
                qml.CNOT(wires=[w, (w + 1) % self.n_wires])
            
            # Final rotation layer
            for w in range(self.n_wires):
                qml.RZ(self.theta_rz[l, w], wires=w)
        
        # Measurements: Pauli-Z expectation values
        paulis = [qml.PauliZ(w) for w in range(self.n_wires)]
        return qml.expval(qml.sum(*paulis))  # Single scalar measurement


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.
        
        Args:
            x: Input tensor of shape (batch_size, num_qubits)
            
        Returns:
            Output tensor of shape (batch_size, 1)  # ← Changed from (batch_size, num_qubits)
        """
        outs = []
        
        # Process each sample in the batch
        for sample in x:
            y = self.circuit(sample)  # Now returns scalar
            
            # Convert to tensor if needed
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            
            outs.append(y)
        
        # Stack and reshape to [batch_size, 1]
        result = torch.stack(outs).unsqueeze(1).to(self.torch_device, dtype=torch.float32)
        
        return result  # [batch_size, 1]