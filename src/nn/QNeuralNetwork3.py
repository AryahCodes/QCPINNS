import pennylane as qml
import torch
import torch.nn as nn
# from qiskit_ionq import IonQProvider  <-- REMOVED

class QNeuralNetwork3(nn.Module):
    """
    IonQ-compatible qubit-based version of CVNeuralNetwork3.
    (REVERTED to old pennylane-qiskit syntax)
    """
    def __init__(
        self,
        num_qumodes: int,
        num_layers: int,
        device: str = "cpu",
        cutoff_dim: int = 2,
        use_cubic_phase: bool = True,
        use_cross_kerr: bool = True,
        learnable_input_encoding: bool = True,
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.device = device
        self.use_cubic_phase = use_cubic_phase
        self.use_cross_kerr = use_cross_kerr
        self.learnable_input_encoding = learnable_input_encoding

        # --- THIS IS THE OLD, QISKIT-PLUGIN SYNTAX ---
        self.dev = qml.device(
            "qiskit.ionq",
            wires=self.num_qumodes,
            backend="ionq_simulator",
            ibmq_api_key="uPshzBiQ04JDnTJBbcEmoCvX1Sf0MToK",
            shots=1000,
        )
        # --- END OF FIX ---

        # Trainable parameters
        active_sd = 1e-4
        passive_sd = 1e-1
        self.theta_1_ry = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * passive_sd
        )
        self.theta_1_rz = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * passive_sd
        )
        self.theta_2_ry = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * passive_sd
        )
        self.theta_2_rz = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * passive_sd
        )
        self.theta_rx = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * active_sd
        )
        self.theta_ry = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * active_sd
        )
        self.theta_rz = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=device) * active_sd
        )
        if use_cross_kerr:
            self.theta_zz = nn.Parameter(
                torch.randn(num_layers, num_qumodes, device=device) * active_sd
            )
        if use_cubic_phase:
            self.theta_cubic_rz = nn.Parameter(
                torch.randn(num_layers, num_qumodes, device=device) * active_sd
            )
        if learnable_input_encoding:
            self.input_scaling = nn.Parameter(torch.ones(num_qumodes, device=device))
            self.input_phase = nn.Parameter(torch.zeros(num_qumodes, device=device))

        self.circuit = qml.QNode(
            self._quantum_circuit, 
            self.dev, 
            interface="torch",
            diff_method="parameter-shift"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(sample) for sample in x])

    def _quantum_circuit(self, inputs):
        for w, val in enumerate(inputs):
            scale = self.input_scaling[w] if self.learnable_input_encoding else 1.0
            phase = self.input_phase[w] if self.learnable_input_encoding else 0.0
            qml.RX(val * scale, wires=w)
            if phase != 0.0:
                qml.RZ(phase, wires=w)

        for l in range(self.num_layers):
            for w in range(self.num_qumodes):
                qml.RY(self.theta_1_ry[l, w], wires=w)
                qml.RZ(self.theta_1_rz[l, w], wires=w)
            for w in range(self.num_qumodes):
                qml.CNOT(wires=[w, (w + 1) % self.num_qumodes])
            for w in range(self.num_qumodes):
                qml.RX(self.theta_rx[l, w], wires=w)
                qml.RY(self.theta_ry[l, w], wires=w)
                qml.RZ(self.theta_rz[l, w], wires=w)
                if self.use_cubic_phase:
                    qml.RZ(self.theta_cubic_rz[l, w], wires=w)
            if self.use_cross_kerr:
                for w in range(self.num_qumodes):
                    qml.IsingZZ(self.theta_zz[l, w], wires=[w, (w + 1) % self.num_qumodes])
            for w in range(self.num_qumodes):
                qml.RY(self.theta_2_ry[l, w], wires=w)
                qml.RZ(self.theta_2_rz[l, w], wires=w)

        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qumodes)]