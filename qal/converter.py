"""Map a data sample x with label y to H_x = I - U(x)^dagger P_y U(x).

U(x) is the data-encoding unitary and P_y the label projector, so
<psi|H_x|psi> is the probability of predicting the wrong label. Three
encodings are used in the paper:

  'classical data'           x is a real vector, encoded by the circuit of
                             build_circuit_encoding_data (Supplementary Information).
  'hamiltonian evolution'    x is a Hamiltonian H and U = exp(-i t H).
  'hamiltonian ground state' x is a Hamiltonian with ground state |g>. A fixed
                             2n-qubit Hamiltonian H_c turns |g> into the n-qubit
                             Hamiltonian (<g| x I) H_c (|g> x I), and U is its
                             evolution for time t.
"""
import cirq
import numpy as np
import scipy.linalg

from .quantum_basics import Hamiltonian, pauli_string_to_matrix


class HConverter:
    def __init__(self, num_qubits, encoding, PVM, t=None, hamiltonian=None):
        self.num_qubits = num_qubits
        self.encoding = encoding
        self.PVM = PVM
        self.t = t
        self.hamiltonian = hamiltonian

    def convert(self, X, y):
        if self.encoding == 'classical data':
            U = cirq.unitary(build_circuit_encoding_data(X, self.num_qubits))
        elif self.encoding == 'hamiltonian evolution':
            U = scipy.linalg.expm(-1j * self.t * X.to_matrix().toarray())
        elif self.encoding == 'hamiltonian ground state':
            ground_state = X.ground_state()
            H = sandwich_hamiltonian(ground_state, self.hamiltonian)
            U = scipy.linalg.expm(-1j * self.t * H.to_matrix().toarray())
        else:
            raise ValueError(f'unknown encoding {self.encoding!r}')
        P = self.PVM.get_projector(y)
        return np.eye(2**self.num_qubits) - U.conj().T @ P @ U


def build_circuit_encoding_data(x, num_qubits, qubits=None):
    """Circuit that encodes the vector x on num_qubits qubits.

    Each layer applies Y, Z, Y rotations to every qubit (three entries of x
    per qubit), then two brick layers of CNOTs and two of CZs. x is
    zero-padded to fill ceil(len(x) / 3n) layers plus floor(n/2) - 1 extra
    layers that only scramble. Rotations by zero are dropped. A rotation
    Y**a is a rotation by angle pi*a.
    """
    if qubits is None:
        qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    depth = (len(x) - 1) // (3 * num_qubits) + (num_qubits // 2)
    N = 3 * num_qubits * depth
    x = np.append(x, np.zeros(N - len(x)))
    for i in range(depth):
        for j in range(num_qubits):
            circuit.append(cirq.Y(qubits[j]) ** x[(3 * i) * num_qubits + j])
            circuit.append(cirq.Z(qubits[j]) ** x[(3 * i + 1) * num_qubits + j])
            circuit.append(cirq.Y(qubits[j]) ** x[(3 * i + 2) * num_qubits + j])
        for j in range(0, num_qubits - 1, 2):
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
        for j in range(1, num_qubits - 1, 2):
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
        for j in range(0, num_qubits - 1, 2):
            circuit.append(cirq.CZ(qubits[j], qubits[j + 1]))
        for j in range(1, num_qubits - 1, 2):
            circuit.append(cirq.CZ(qubits[j], qubits[j + 1]))
    # with_noise is used here only as a convenient way to rewrite every operation
    return circuit.with_noise(EliminateZeroRotations(num_qubits))


def sandwich_hamiltonian(psi, H: Hamiltonian):
    """Return the m-qubit Hamiltonian (<psi| x I) H (|psi> x I) for an n-qubit state psi and an (n+m)-qubit H."""
    n = int(np.log2(len(psi)) + 0.1)
    m = int(H.num_qubits - n)
    terms = {}
    for pauli_string, coeff in H.terms.items():
        pauli_1 = pauli_string[:n]
        pauli_2 = pauli_string[n:]
        terms[pauli_2] = terms.get(pauli_2, 0) + coeff * np.vdot(psi, pauli_string_to_matrix(pauli_1) @ psi)
    return Hamiltonian(m, terms)


class EliminateZeroRotations(cirq.NoiseModel):
    """Drop X/Y/Z power gates with a zero exponent (the padding of the encoding)."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def noisy_operation(self, operation):
        if isinstance(operation.gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)):
            if np.isclose(operation.gate.exponent, 0):
                return []
        return operation
