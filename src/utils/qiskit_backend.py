from typing import Optional

def build_qiskit_backend(provider: str = "", backend_name: Optional[str] = None, token: Optional[str] = None):
    """Return a Qiskit backend instance for use with pennylane-qiskit 'qiskit.remote' device.

    Supported providers:
    - "ionq": requires qiskit-ionq installed and an API token. Backend examples:
        "ionq_simulator", "ionq_qpu_aria-2", "ionq_qpu_forte-1" (names subject to provider list)
    - "ibm": requires qiskit_ibm_runtime configured (token via QiskitRuntimeService.save_account)

    If backend_name is None, the provider's default/simulator may be used.
    """
    provider = (provider or "").lower()
    if provider == "ionq":
        # This code uses the MODERN qiskit_ionq, but we will call it from
        # the OLD pennylane-qiskit. This is a bit advanced, but it's how your friend does it.
        from qiskit_ionq import IonQProvider

        if token is None:
            import os

            token = os.getenv("IONQ_API_TOKEN", "uPshzBiQ04JDnTJBbcEmoCvX1Sf0MToK") # I've added your key as a fallback
            if not token:
                raise ValueError("IONQ_API_TOKEN is not set; please export it or pass token explicitly.")
        ionq = IonQProvider(token)
        
        if backend_name is None:
            backend_name = "ionq_simulator"  # Default to simulator
        return ionq.get_backend(backend_name)

    if provider == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        if backend_name is None:
            return service.least_busy(operational=True, simulator=False)
        return service.backend(backend_name)

    raise ValueError(f"Unsupported qiskit provider: {provider}")