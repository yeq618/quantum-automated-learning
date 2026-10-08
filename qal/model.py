"""The QAL model: pure-state training and evaluation, and the noisy density-matrix version."""
import multiprocessing as mp

import cirq
import numpy as np

from .converter import build_circuit_encoding_data
from .quantum_basics import CanonicalPVM


class QuantumAutomatedLearningModel:
    """Pure-state QAL model. The quantum state plays the role of the trainable parameters.

    The state starts in a random computational basis state. A training step
    applies the post-selected update |psi> <- (I - lr H_x)|psi> / norm, and
    succ_prob accumulates the product of the post-selection probabilities.
    """

    def __init__(self, num_qubits, num_classes):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.I = np.eye(2**num_qubits)
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[np.random.randint(2**num_qubits)] = 1
        self.succ_prob = 1.0

    def train(self, H, lr):
        state = (self.I - lr * H) @ self.state
        self.succ_prob = self.succ_prob * np.linalg.norm(state) ** 2
        self.state = state / np.linalg.norm(state)

    def predict(self, H):
        """Measure the label of the sample with matrix H and return whether the prediction is correct.

        The prediction is wrong with probability ||H psi||^2, and the state
        collapses to the matching post-measurement state (binary classification).
        """
        wrong_prob = np.linalg.norm(H @ self.state) ** 2
        if np.random.rand() < wrong_prob:
            is_correct = False
            state = H @ self.state
        else:
            is_correct = True
            state = (self.I - H) @ self.state
        self.state = state / np.linalg.norm(state)
        return is_correct


def _accuracy_worker(args):
    state, H_path = args
    H = np.load(H_path)
    return 1 - np.linalg.norm(H @ state) ** 2


def evaluation(model, loading_dataset, num_workers):
    """Probability of predicting the correct label, for every sample of a LoadingDataset."""
    args = [(model.state, path) for path in loading_dataset.paths]
    with mp.Pool(num_workers) as pool:
        return pool.map(_accuracy_worker, args)


class MixedQuantumAutomatedLearningModel:
    """Density-matrix QAL model with gate noise (Fig. 2e).

    A training step simulates U(x), the label-dependent rotation of an
    ancilla, and U(x)^dagger with a noisy density-matrix simulator, then
    post-selects the ancilla on |0>. The state starts maximally mixed.
    """

    def __init__(self, num_qubits, num_classes, noise_model):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.noise_model = noise_model
        self.qubits = cirq.LineQubit.range(num_qubits)
        self.ancilla = cirq.NamedQubit('ancilla')
        self.PVM = CanonicalPVM(num_qubits, num_classes)
        self.state = np.eye(2**num_qubits) / 2**num_qubits
        self.succ_prob = 1.0

    def train(self, data, lr):
        X, y = data
        encoding_circuit = build_circuit_encoding_data(X, self.num_qubits, self.qubits)
        C = cirq.Circuit()
        C.append(encoding_circuit)
        C.append(self.PVM.get_circuit(self.qubits, self.ancilla, y, lr))
        C.append(cirq.inverse(encoding_circuit))
        simulator = cirq.DensityMatrixSimulator(noise=self.noise_model)
        result = simulator.simulate(C, initial_state=np.kron(self.state, np.array([[1, 0], [0, 0]])))
        output_state = result.final_density_matrix
        # project the ancilla (last qubit) onto |0>
        output_state = np.kron(np.eye(2**self.num_qubits), np.array([[1, 0]])) @ output_state @ np.kron(np.eye(2**self.num_qubits), np.array([[1], [0]]))
        prob = np.trace(output_state).real
        self.succ_prob = self.succ_prob * prob
        output_state = output_state / prob
        output_state = (output_state + output_state.conj().T) / 2
        self.state = output_state


def _mixed_accuracy_worker(args):
    state, data, noise_model, PVM = args
    X, y = data
    encoding_circuit = build_circuit_encoding_data(X, PVM.num_qubits)
    simulator = cirq.DensityMatrixSimulator(noise=noise_model)
    result = simulator.simulate(encoding_circuit, initial_state=state)
    output_state = result.final_density_matrix
    projector = PVM.get_projector(y)
    return np.trace(projector @ output_state).real


def mixed_evaluation(model, dataset, num_workers):
    """Probability of predicting the correct label under noise, for every sample of a ClassicalDataset."""
    args = [(model.state, dataset.get_sample(i), model.noise_model, model.PVM) for i in range(dataset.len)]
    with mp.Pool(num_workers) as pool:
        return pool.map(_mixed_accuracy_worker, args)
