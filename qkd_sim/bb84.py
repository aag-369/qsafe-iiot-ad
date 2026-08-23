"""
BB84 Quantum Key Distribution simulation using IBM Qiskit.

Each call to `simulate_bb84_round` builds one Qiskit circuit containing
`n_qubits` independent BB84 rounds (one qubit per round), batched into a
single circuit for efficient execution on Aer. For every qubit:

  1. Alice picks a random bit and a random basis (Z or X) and prepares the
     qubit accordingly.
  2. Optionally, an eavesdropper (Eve) intercepts the qubit: she measures in
     a random basis of her own, then re-prepares and forwards a qubit
     encoding her measured bit in her (possibly wrong) basis. This is the
     textbook intercept-resend attack, the mechanism underlying "harvest
     now, decrypt later" reconnaissance against the quantum channel.
  3. The channel itself is modeled with a depolarizing noise channel to
     represent benign photonic loss / decoherence.
  4. Bob picks his own random basis and measures.

After the round, Alice and Bob publicly compare bases (sifting) and the
Quantum Bit Error Rate (QBER) is computed over the sifted key: the fraction
of sifted bits where Alice's and Bob's values disagree.

This module produces one QBER sample per call. `qber_stream.py` calls it
repeatedly to build a time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Reused across calls; building an AerSimulator has fixed setup cost.
# method="automatic" with no coupling map lets Aer simulate all-to-all
# connectivity, which is what we want here: there is no hardware topology
# constraint in a QKD channel simulation, only gate-level physics.
_SIMULATOR = AerSimulator(method="automatic")


@dataclass
class BB84Result:
    qber: float
    sifted_key_length: int
    n_qubits: int
    n_intercepted: int


def _build_noise_model(channel_error_prob: float) -> NoiseModel:
    """Depolarizing single-qubit noise standing in for photonic loss/decoherence."""
    noise_model = NoiseModel()
    if channel_error_prob > 0:
        error = depolarizing_error(channel_error_prob, 1)
        noise_model.add_all_qubit_quantum_error(error, ["id", "x", "h"])
    return noise_model


def simulate_bb84_round(
    n_qubits: int,
    channel_error_prob: float = 0.01,
    eve_intercept_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> BB84Result:
    """Simulate one batch of `n_qubits` independent BB84 exchanges.

    Args:
        n_qubits: number of BB84 rounds (qubits) to batch into one circuit.
        channel_error_prob: benign depolarizing error probability per qubit,
            representing environmental noise / photonic loss.
        eve_intercept_prob: probability that any given qubit is intercepted
            and resent by an eavesdropper using the intercept-resend attack.
            0.0 means no attack is present in this round.
        rng: optional numpy random Generator for reproducibility.

    Returns:
        BB84Result with the measured QBER and bookkeeping counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, size=n_qubits)
    alice_bases = rng.integers(0, 2, size=n_qubits)  # 0 = Z basis, 1 = X basis
    bob_bases = rng.integers(0, 2, size=n_qubits)

    intercepted = rng.random(n_qubits) < eve_intercept_prob
    eve_bases = rng.integers(0, 2, size=n_qubits)

    qreg = QuantumRegister(n_qubits, "q")
    creg = ClassicalRegister(n_qubits, "c")
    # A second classical register captures Eve's intermediate measurement
    # when she intercepts, so her resend can be conditioned on it.
    eve_creg = ClassicalRegister(n_qubits, "eve_c")
    circuit = QuantumCircuit(qreg, creg, eve_creg)

    # --- Alice's state preparation ---
    for i in range(n_qubits):
        if alice_bits[i] == 1:
            circuit.x(qreg[i])
        if alice_bases[i] == 1:  # X basis
            circuit.h(qreg[i])

    # --- Eve's intercept-resend (only on intercepted qubits) ---
    intercepted_idx = np.where(intercepted)[0]
    for i in intercepted_idx:
        if eve_bases[i] == 1:
            circuit.h(qreg[i])
        circuit.measure(qreg[i], eve_creg[i])
        circuit.reset(qreg[i])
        # Re-prepare based on Eve's measured bit, in Eve's chosen basis.
        with circuit.if_test((eve_creg[i], 1)):
            circuit.x(qreg[i])
        if eve_bases[i] == 1:
            circuit.h(qreg[i])

    # --- Bob's measurement ---
    for i in range(n_qubits):
        if bob_bases[i] == 1:
            circuit.h(qreg[i])
        circuit.measure(qreg[i], creg[i])

    noise_model = _build_noise_model(channel_error_prob)
    job = _SIMULATOR.run(
        circuit,
        shots=1,
        noise_model=noise_model,
        memory=True,
    )
    result = job.result()
    memory = result.get_memory()[0]  # bitstring, creg 'c' is rightmost block

    # Qiskit's classical bitstring is space-separated per register, MSB first,
    # ordered as declared in reverse: "eve_c c" -> "<eve_c bits> <c bits>".
    parts = memory.split(" ")
    bob_bits_str = parts[-1]
    bob_bits = np.array([int(b) for b in bob_bits_str[::-1]])  # index 0 = qubit 0

    # --- Sifting: keep only rounds where Alice and Bob chose the same basis ---
    sifted_mask = alice_bases == bob_bases
    sifted_alice = alice_bits[sifted_mask]
    sifted_bob = bob_bits[sifted_mask]

    sifted_len = int(sifted_mask.sum())
    if sifted_len == 0:
        qber = 0.0
    else:
        errors = int(np.sum(sifted_alice != sifted_bob))
        qber = errors / sifted_len

    return BB84Result(
        qber=qber,
        sifted_key_length=sifted_len,
        n_qubits=n_qubits,
        n_intercepted=int(intercepted.sum()),
    )
