#Importing relevant libraries:::
import numpy as np
import qiskit
from qiskit import Aer, QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier



#load in preprocessed data 
#paths to data...
train_images_path = "train_images.npy"
train_labels_path = "train_labels.txt"

train_images = np.load(train_images_path)
train_labels = np.load(train_labels_path)

#splitting data...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#QUANTUM ENCODING :::

#encoding using Ry rotationla gates here
#classical data is scaled to [0, pi] and encoded into qubits [with Ry gates]
def quantum_encoding(circuit, features):
    for i, x in enumerate(features):
        circuit.ry(x, i)



#QUANTUM LAYER W ENTANGLEMENT :::
#... and parametrized Ry(θ) gates = params - > makes data trainable for teh qunatum layer.
#entanglement = CNOT gates between qubits =  circuit.cx(i,i+1)
def quantum_layer(circuit, params):
    num_qubits = circuit.num_qubits
    for i in range(num_qubits):
        circuit.ry(params[i], i)  
    for i in range(num_qubits - 1):  
        circuit.cx(i, i + 1)


#LEARNABLE QUANTUM CIRCUIT:::

#func to define number of qubits and the quantum circuit.
#calling function to encoding layer to add encoding into the quantum circuit
#calling the function of parametrized quantum layer 
#measuring and returning the circuit.
def create_qcnn_circuit(features, params):
    num_qubits = len(features)
    circuit = QuantumCircuit(num_qubits)

    quantum_encoding(circuit, features)
    
    quantum_layer(circuit, params)
    
    circuit.measure_all()
    return circuit



#adding in a func here to simulate quantum circuit
#calculates value <Z> 
#copmutes parity
def run_circuit(circuit):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator, shots=1024).result()
    counts = result.get_counts()
    
    
    total_shots = sum(counts.values())
    z_expectation = 0
    for state, count in counts.items():
        parity = sum(int(bit) for bit in state) % 2
        z_expectation += (-1)**parity * count / total_shots
    return z_expectation




#HYBRID MODEL BUILD:::
#combining quantum and classical elements here
#quantum ouputs = z_expectation are fed into the CNN 
def hybrid_model(X, params):
    quantum_outputs = []
    for features in X:
        circuit = create_qcnn_circuit(features, params)
        quantum_outputs.append(run_circuit(circuit))
    return np.array(quantum_outputs).reshape(-1, 1)





#TRAIN MODEL:::
#clssical parameters are optimized, vs quantum outputs = remain the same.
#intitialising variables in the quantum circuit
num_qubits = X_train.shape[1]
initial_params = np.random.rand(num_qubits)

#generating quantum outputs
X_train_quantum = hybrid_model(X_train, initial_params)
X_test_quantum = hybrid_model(X_test, initial_params)

#training mlp classifier w quantum data
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=42)
clf.fit(X_train_quantum, y_train)




#[[OTHER FILE FOROUTPUTS AND EVALUATION]]!!!
#but for now here:
accuracy = clf.score(X_test_quantum, y_test)
print(f"Hybrid QCNN Accuracy: {accuracy * 100:.2f}%")