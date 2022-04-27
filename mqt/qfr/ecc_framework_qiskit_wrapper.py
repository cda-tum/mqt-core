#!/bin/python3

import sys
import numpy as np
from qiskit import execute, QuantumCircuit
from mqt import qfr
import argparse

# Qiskit Aer noise module imports
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import kraus_error
from qiskit import Aer
from qiskit import IBMQ
import qiskit


def compose_error(error, new_error):
    if error is None:
        error = new_error
    else:
        error = error.compose(new_error)
    return error


def create_noise_model(n_model, p_error):
    # Create an empty noise model
    noise_model = NoiseModel()
    error = None
    for char in n_model:
        # print("Adding noise model for " + char)
        if char == 'B':
            new_error = pauli_error([('X', p_error), ('I', 1 - p_error)])
            error = compose_error(error, new_error)
        elif char == 'D':
            # Add a depolarization error channel
            new_error = depolarizing_error(p_error, 1)
            error = compose_error(error, new_error)
        elif char == 'A':
            # add a simple amplitude damping error channel which mimics energy loss to the environment (T1-error)
            ap_error = p_error
            A0 = np.array([[1, 0], [0, np.sqrt(1 - 2 * ap_error)]], dtype=complex)
            A1 = np.array([[0, np.sqrt(2 * ap_error)], [0, 0]], dtype=complex)
            noise_ops = [a for a in [A0, A1] if np.linalg.norm(a) > 1e-10]
            new_error = kraus_error(noise_ops, canonical_kraus=True)
            error = compose_error(error, new_error)
        elif char == 'Y':
            # Generalized amplitude damping noise
            Y0 = np.sqrt(p_error) * np.array([[1, 0], [0, np.sqrt(1 - 2 * p_error)]], dtype=complex)
            Y1 = np.sqrt(p_error) * np.array([[0, np.sqrt(2 * p_error)], [0, 0]], dtype=complex)
            Y2 = np.sqrt(1 - p_error) * np.array([[np.sqrt(1 - 2 * p_error), 0], [0, 1]], dtype=complex)
            Y3 = np.sqrt(1 - p_error) * np.array([[0, 0], [np.sqrt(2 * p_error), 0]], dtype=complex)
            noise_ops = [a for a in [Y0, Y1, Y2, Y3] if np.linalg.norm(a) > 1e-10]
            new_error = kraus_error(noise_ops, canonical_kraus=True)
            error = compose_error(error, new_error)
        elif char == 'Q':
            new_error = amplitude_damping_error(p_error, 1)
            error = compose_error(error, new_error)
        elif char == 'P':
            new_error = pauli_error([('Z', p_error), ('I', 1 - p_error)])
            error = compose_error(error, new_error)

        else:
            print("Warning unknown error")
    assert error is not None

    noise_model.add_all_qubit_quantum_error(error,
                                            ['u1', 'u2', 'u3', 'h', 'id', 't', 'tdg', 'sdg', 'rx', 'ry', 'rz', 's'])
    noise_model.add_all_qubit_quantum_error(error.tensor(error), ['cx', 'swap'])
    noise_model.add_all_qubit_quantum_error(error.tensor(error).tensor(error), ['cswap'])

    return noise_model


def main():
    parser = argparse.ArgumentParser(description='QiskitWrapper interface with ecc support!')
    parser.add_argument('-m', type=str, default="D", help='Define the noise model (AQYDP) (Default="D")')
    parser.add_argument('-p', type=float, default=0.001, help='Set the noise probability (Default=0.001)')
    parser.add_argument('-n', type=int, default=2000, help='Set the number of shots. 0 for deterministic simulation ('
                                                           'Default=2000)')
    parser.add_argument('-s', type=int, default=0, help='Set a seed (Default=0)')
    parser.add_argument('-f', type=str, required=True, help='Path to openqasm file')
    parser.add_argument('-e', type=str, required=False, default=None,
                        help='Export circuit with ecc as openqasm circuit instead of simulation it (provide name)')
    parser.add_argument('-fs', type=str, default='none',
                        help='Specify a simulator (Default: "statevector_simulator" for simulation without noise, '
                             '"aer_simulator_density_matrix", for deterministic noise-aware simulation'
                             '"aer_simulator_statevector", for stochastic noise-aware simulation). Available: ' + str(
                            Aer.backends()))
    parser.add_argument('-ecc', type=str, default='none',
                        help='Specify a ecc to be applied to the circuit (Default=none)')
    parser.add_argument('-fq', type=int, default=100, help='Set the frequency for error correction (Default=100)')
    parser.add_argument('-mc', type=bool, default=False, help='Only allow single controlled gates (Default=False)')
    parser.add_argument('-cf', type=bool, default=False, help='Only allow clifford operations (Default=False)')

    args = parser.parse_args()

    model = args.m
    err_p = args.p
    stoch_runs = args.n
    seed = args.s
    file = args.f

    if args.fs.lower() == 'none':
        forced_simulator = None
    else:
        forced_simulator = args.fs

    if args.ecc.lower() == 'none':
        ecc = None
    else:
        ecc = args.ecc

    ecc_frequency = args.fq
    ecc_mc = args.mc
    ecc_cf = args.cf
    ecc_export = args.e

    n_shots = 0
    if stoch_runs > 0:
        n_shots = stoch_runs
    else:
        n_shots = 100000

    if err_p > 0:
        noise_model = create_noise_model(n_model=model, p_error=err_p)
    else:
        noise_model = NoiseModel()

    circ = None
    if ecc is not None:
        result = qfr.apply_ecc(file, ecc, ecc_frequency, ecc_mc, ecc_cf)
        if "error" in result:
            print("Something went wrong when I tried to apply the ecc. Error message:\n" + result["error"])
            exit(1)
        circ = QuantumCircuit().from_qasm_str(result["circ"])
    else:
        circ = QuantumCircuit().from_qasm_file(file)

    if ecc_export is not None:
        print("Exporting circuit to: " + str(ecc_export))
        circ.qasm(filename=ecc_export)
        exit(0)

    size = circ.num_qubits
    result_counts = None
    simulator_backend = None
    print("_____Trying to simulate with " + str(model) + "(prob=" + str(err_p) + ", shots=" + str(
        n_shots) + ", n_qubits=" + str(size) + ") Error______", flush=True)

    if forced_simulator is not None:
        # Setting the simulator backend to the requested one
        try:
            simulator_backend = Aer.get_backend(forced_simulator)
        except qiskit.providers.exceptions.QiskitBackendNotFoundError:
            print("Unknown backend specified.\nAvailable backends are " + str(Aer.backends()))
            exit(22)
    elif err_p == 0:
        # Statevector simulation method
        simulator_backend = Aer.get_backend('statevector_simulator')
    elif stoch_runs == 0:
        # Run the noisy density matrix (deterministic) simulation
        simulator_backend = Aer.get_backend('aer_simulator_density_matrix')
    else:
        # Stochastic statevector simulation method
        simulator_backend = Aer.get_backend('aer_simulator_statevector')

    result = execute(circ,
                     backend=simulator_backend,
                     shots=n_shots,
                     seed_simulator=seed,
                     noise_model=noise_model,
                     optimization_level=0)
    try:
        result_counts = result.result().get_counts()
    except:
        print("Simulation exited with status: " + str(result.result().status))
        exit(1)

    delme = result.result()

    printed_results = 0

    summarized_counts = dict()
    for result_id in result_counts:
        sub_result = result_id.split(' ')[-1]
        if sub_result not in summarized_counts.keys():
            summarized_counts[sub_result] = 0
        summarized_counts[sub_result] += result_counts[result_id]

    for result_id in sorted(summarized_counts.keys()):
        # if result_counts[result_id] / n_shots > 0.001 or printed_results == 0:
        if summarized_counts[result_id] / n_shots > 0 or printed_results == 0:  # Print all results > 0
            result_string = str(result_id)
            print("State |" + result_string + "> probability " + str(summarized_counts[result_id] / n_shots))
            printed_results += 1
            if printed_results == 1000:
                break


if __name__ == "__main__":
    main()
