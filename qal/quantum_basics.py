"""Pauli operators, model Hamiltonians, the label measurement and the noise model."""
import random

import cirq
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

pauli_I = sp.eye(2, format='csc')
pauli_X = sp.csc_matrix([[0, 1], [1, 0]])
pauli_Y = sp.csc_matrix([[0, -1j], [1j, 0]])
pauli_Z = sp.csc_matrix([[1, 0], [0, -1]])
pauli_dict = {'I': pauli_I, 'X': pauli_X, 'Y': pauli_Y, 'Z': pauli_Z}


def pauli_string_to_matrix(pauli_string):
    """Sparse matrix of a Pauli string such as 'XIZ'. The first character acts on the first qubit."""
    if len(pauli_string) == 1:
        return pauli_dict[pauli_string]
    return sp.kron(pauli_dict[pauli_string[0]], pauli_string_to_matrix(pauli_string[1:]))


class Hamiltonian:
    """H = sum_P c_P P, stored as a dict from Pauli strings to coefficients."""

    def __init__(self, num_qubits, terms):
        self.num_qubits = num_qubits
        self.terms = terms

    def to_matrix(self):
        H = sp.csc_matrix((2**self.num_qubits, 2**self.num_qubits))
        for pauli_string, coeff in self.terms.items():
            H = H + coeff * pauli_string_to_matrix(pauli_string)
        return H

    def ground_state(self):
        # A fixed starting vector makes the result independent of ARPACK's internal random state.
        v0 = np.random.default_rng(0).standard_normal(2**self.num_qubits)
        eigen = sp.linalg.eigsh(self.to_matrix(), k=1, which='SA', v0=v0)
        return eigen[1][:, 0]


class RandomHamiltonian(Hamiltonian):
    """Random (n+m)-qubit Hamiltonian with t terms.

    Each term is a product of a random 2-body Pauli string on the first n
    qubits and a random 2-body Pauli string on the last m qubits, with a
    coefficient drawn uniformly from [-1, 1]. The fixed Hamiltonian in
    data/cim_converter_hamiltonian.json was drawn as RandomHamiltonian(10, 10, 100).
    """

    def __init__(self, n, m, t):
        terms = {}
        for _ in range(t):
            pauli_string1 = self.generate_random_pauli(n, 2)
            pauli_string2 = self.generate_random_pauli(m, 2)
            coeff = np.random.rand() * 2 - 1
            terms[pauli_string1 + pauli_string2] = coeff
        super().__init__(n + m, terms)

    @staticmethod
    def generate_random_pauli(n, k):
        pauli_list = ['I'] * n
        non_I_indices = random.sample(range(n), k)
        pauli_choices = random.choices(['X', 'Y', 'Z'], k=k)
        for i, j in zip(non_I_indices, pauli_choices):
            pauli_list[i] = j
        return ''.join(pauli_list)


class ClusteringIsingHamiltonian(Hamiltonian):
    """H = -sum_i X_{i-1} Z_i X_{i+1} + l sum_i Y_i Y_{i+1}, periodic boundary conditions.

    The ground state is in a symmetry-protected topological phase for l < 1
    and in an antiferromagnetic phase for l > 1.
    """

    def __init__(self, num_qubits, l):
        terms = {}
        for i in range(num_qubits - 2):
            terms['I' * i + 'XZX' + 'I' * (num_qubits - i - 3)] = -1
        for i in range(num_qubits - 1):
            terms['I' * i + 'YY' + 'I' * (num_qubits - i - 2)] = l
        # terms that close the ring
        terms['X' + 'I' * (num_qubits - 3) + 'XZ'] = -1
        terms['ZX' + 'I' * (num_qubits - 3) + 'X'] = -1
        terms['Y' + 'I' * (num_qubits - 2) + 'Y'] = l
        super().__init__(num_qubits, terms)


class AubryAndreHamiltonian(Hamiltonian):
    """H = -g/2 sum_k (X_k X_{k+1} + Y_k Y_{k+1}) - sum_k V_k/2 Z_k, open boundary conditions.

    V_k = V cos(2 pi alpha k) with alpha = (sqrt(5) - 1) / 2 and g = 1. The
    model is delocalized for V < 2 and localized for V > 2 (arXiv:2204.01738, Eq. 3).
    """

    def __init__(self, num_qubits, V):
        g = 1.0
        phi = 0.0
        alpha = (np.sqrt(5) - 1) / 2
        terms = {}
        for i in range(num_qubits - 1):
            terms['I' * i + 'XX' + 'I' * (num_qubits - i - 2)] = -g / 2
            terms['I' * i + 'YY' + 'I' * (num_qubits - i - 2)] = -g / 2
        for i in range(num_qubits):
            terms['I' * i + 'Z' + 'I' * (num_qubits - i - 1)] = -V / 2 * np.cos(2 * np.pi * alpha * i + phi)
        super().__init__(num_qubits, terms)


class CanonicalPVM:
    """Label measurement on the middle qubit(s).

    For k classes, N = ceil(log2 k) qubits in the middle of the register carry
    the label, and P_y projects them onto the computational basis state |y>.
    """

    def __init__(self, num_qubits, num_classes):
        self.num_qubits = num_qubits
        self.num_classes = num_classes

    def label_qubits(self):
        """Number N of label qubits and the index a of the first one."""
        N = 1
        while 2**N < self.num_classes:
            N += 1
        return N, (self.num_qubits - N) // 2

    def get_projector(self, label):
        N, a = self.label_qubits()
        b = self.num_qubits - N - a
        diag = np.zeros(2**N, dtype=np.complex128)
        diag[label] = 1
        projector = np.diag(diag)
        return np.kron(np.eye(2**a), np.kron(projector, np.eye(2**b)))

    def get_circuit(self, qubits, ancilla, label, lr):
        """Circuit for the non-unitary step P_y + (1 - lr) sum_{y' != y} P_y'.

        A controlled R_y rotates the ancilla when the label qubit differs from
        y. Post-selecting the ancilla on |0> then applies the operator above.
        Only binary classification is implemented.
        """
        assert self.num_classes == 2
        theta = 2 * np.arccos(1 - lr)
        controlled_Ry = cirq.ControlledGate(cirq.ry(theta), num_controls=1)
        middle = qubits[self.label_qubits()[1]]
        C = cirq.Circuit()
        if label == 1:
            C.append(cirq.X(middle))
        C.append(controlled_Ry(middle, ancilla))
        if label == 1:
            C.append(cirq.X(middle))
        return C


class DepolarizingNoiseModel(cirq.NoiseModel):
    """Depolarizing channel after every one- and two-qubit gate (density-matrix simulation)."""

    def __init__(self, single_qubit_noise_rate: float, two_qubit_noise_rate: float):
        self.single_qubit_noise = cirq.depolarize(single_qubit_noise_rate, n_qubits=1)
        self.two_qubit_noise = cirq.depolarize(two_qubit_noise_rate, n_qubits=2)

    def noisy_operation(self, operation):
        if isinstance(operation.gate, cirq.Gate):
            qubits = operation.qubits
            if len(qubits) == 1:
                return [operation, self.single_qubit_noise.on(*qubits)]
            elif len(qubits) == 2:
                return [operation, self.two_qubit_noise.on(*qubits)]
        return [operation]
