import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
# from qiskit_ionq import IonQProvider  <-- REMOVED

class QNeuralNetwork2(nn.Module):
    """
    IonQ-compatible qubit-based version of CVNeuralNetwork2.
    (REVERTED to old pennylane-qiskit syntax)
    """

    def __init__(self, num_qumodes: int, num_layers: int, device: str = "cpu"):
        super().__init__()
        self.n_wires = num_qumodes
        self.num_layers = num_layers
        self.torch_device = device

        # --- THIS IS THE OLD, QISKIT-PLUGIN SYNTAX ---
        self.dev = qml.device(
            "qiskit.ionq",
            wires=self.n_wires,
            backend="ionq_simulator",
            ibmq_api_key="uPshzBiQ04JDnTJBbcEmoCvX1Sf0MToK",
            shots=1000,
        )
        # --- END OF FIX ---

        # Trainable parameters
        active_sd = 0.1
        passive_sd = 2 * np.pi
        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, self.n_wires, device=self.torch_device) * passive_sd,
            requires_grad=True,
        )
        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, self.n_wires, device=self.torch_device) * passive_sd,
            requires_grad=True,
        )
        self.theta_rx = nn.Parameter(
            torch.randn(num_layers, self.n_wires, device=self.torch_device) * active_sd,
            requires_grad=True,
        )
        self.theta_ry = nn.Parameter(
            torch.randn(num_layers, self.n_wires, device=self.torch_device) * active_sd,
            requires_grad=True,
        )
        self.theta_rz = nn.Parameter(
            torch.randn(num_layers, self.n_wires, device=self.torch_device) * active_sd,
            requires_grad=True,
        )

        self.circuit = qml.QNode(
            self._quantum_circuit, 
            self.dev, 
            interface="torch",
            diff_method="parameter-shift"
        )
        self.activation = nn.Tanh()

    def _quantum_circuit(self, inputs):
        for w, x in enumerate(inputs):
            qml.RX(x, wires=w)

        for l in range(self.num_layers):
            for w in range(self.n_wires):
                qml.RY(self.theta_1[l, w], wires=w)
            for w in range(self.n_wires):
                qml.CNOT(wires=[w, (w + 1) % self.n_wires])
            for w in range(self.n_wires):
                qml.RX(self.theta_rx[l, w], wires=w)
                qml.RY(self.theta_ry[l, w], wires=w)
                qml.RZ(self.theta_rz[l, w], wires=w)
            for w in range(self.n_wires):
                qml.RY(self.theta_2[l, w], wires=w)

        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(sample) for sample in x])