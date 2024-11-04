from qiskit import QuantumCircuit

def my_entangling(n_qubits=2, measure=False):
    qc = QuantumCircuit(n_qubits, n_qubits, name="my_entangling")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(0, i + 1)
    if measure == True:
        qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc


def my_superposition(n_qubits=1, measure=False):
    qc = QuantumCircuit(n_qubits, n_qubits, name="my_superposition")
    for i in range(n_qubits):
        qc.h(i)
    if measure == True:
        qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc