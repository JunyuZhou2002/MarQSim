'''
MarQSim Compiler Script

This script implements the MarQSim framework, a hybrid quantum compiler that unifies deterministic 
and randomized compilation approaches for quantum Hamiltonian simulation. It constructs and 
optimizes transition matrices using Markov chain sampling, minimum-cost flow models, and stochastic 
perturbations to improve gate efficiency in quantum circuit synthesis.

Key functionalities include:
- 1. Parsing Pauli string Hamiltonians.
- 2. Generating various Markov transition matrices (Algorithm 2 from MarQSim).
- 3. Compiling Hamiltonian evolution circuits via controlled sampling (Algorithm 1 from MarQSim).

This code supports empirical evaluation of the MarQSim framework, including reproducible 
experiments for the results presented in the MarQSim paper (PLDI 2025).
'''

import os
import sys
import csv
import copy
import time
import torch
import argparse
import pandas as pd
import numpy as np
from numpy import mat
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file = sys.stdout

'''
Set the computation device to GPU (CUDA) for faster processing
Set the default tensor data type to 64-bit float for higher precision
Set the seed for NumPy's random number generator to ensure reproducibility
'''
device = 'cuda' 
torch.set_default_dtype(torch.float64)      
np.random.seed(2012)

############################### Hamiltonian Parsing ###############################

def get_pauli(path: str):
    '''
    This function extracts Pauli string information from a file at the given path.
    The input file is expected to have lines like:
        - 0.7555182173201127 * IIIIIIIZ
        - 0.9492679550589105 * IIIIIIZI
    The output is a list of [Pauli_string, coefficient] pairs, e.g.:
        [['IIIIIIZI', -0.9493], ...]
    '''
    file = open(file=path, mode="r", encoding='utf-8')
    file = file.readlines()
    res = []
    for line in file:
        if len(line) < 2:
            continue
        line = line.split(' ')
        val = int(float(line[1]) * 10000 + 0.5)
        if val == 0:
            continue
        val = float(val) / 10000
        if line[0] == '+':
            res.append([line[3][:-1], val])
        if line[0] == '-':
            res.append([line[3][:-1], -val])
    return res

def get_hamiltonian(list):
    '''
    Given an input like ['IIIIIIZI', -0.9493], this function computes the matrix form of w_j * P_j,
    where:
        - w_j = list[1] is a scalar coefficient,
        - P_j is the tensor product of Pauli matrices defined by the string list[0].
    The result is the full matrix representation of the weighted Pauli operator.
    '''
    res = torch.tensor([[1.0 + 0.0j]], device=device)
    op_I = torch.tensor([[1.0 + 0.0j, 0.0 + 0.j], [0.0 + 0.j, 1.0 + 0.0j]], device=device)
    op_X = torch.tensor([[0.0 + 0.0j, 1.0 + 0.j], [1.0 + 0.j, 0.0 + 0.0j]], device=device)
    op_Y = torch.tensor([[0.0 + 0.0j, 0.0 - 1.j], [0.0 + 1.j, 0.0 + 0.0j]], device=device)
    op_Z = torch.tensor([[1.0 + 0.0j, 0.0 + 0.j], [0.0 + 0.j, -1.0 + 0.0j]], device=device)
    for idx, op in enumerate(list[0]):
        if op == 'I':
            res = torch.kron(res, op_I)
        elif op == 'X':
            res = torch.kron(res, op_X)
        elif op == 'Y':
            res = torch.kron(res, op_Y)
        elif op == 'Z':
            res = torch.kron(res, op_Z)
    return list[1] * res

def get_CNOT_matrix(input_pauli):
    '''
    Given `input_pauli` as the list of Pauli strings (output from get_pauli),
    this function computes a matrix M where:
        M[i][j] = the estimated number of CNOT gates needed to transform Pauli string i into string j.
    '''
    string_num = len(input_pauli)
    string_len = len(input_pauli[0][0])
    res = [[0 for i in range(string_num)] for i in range(string_num)]
    for ii in range(string_num):
        for jj in range(string_num):
            count = 0
            string_i = input_pauli[ii][0]
            string_j = input_pauli[jj][0]
            for i in range(string_len):
                if string_i[i] != string_j[i]:
                    count = count + 1
                    if string_i[i] != 'I' and string_j[i] != 'I':
                        count = count + 1
            res[ii][jj] = count
    return res

def get_single_q_matrix(input_pauli):
    '''
    Given `input_pauli` as a list of Pauli strings (output from get_pauli),
    this function computes a matrix M where:
        M[i][j] = estimated number of single-qubit gates needed to transform Pauli string i into string j.
    '''
    string_num = len(input_pauli)
    string_len = len(input_pauli[0][0])
    res = [[0 for i in range(string_num)] for i in range(string_num)]
    XY_count = [0 for i in range(string_num)]
    for ii in range(string_num):
        string_i = input_pauli[ii][0]
        for i in range(string_len):
            if string_i[i] == 'X':
                XY_count[ii] += 1
            if string_i[i] == 'Y':
                XY_count[ii] += 1
    for ii in range(string_num):
        for jj in range(string_num):
            count = 0
            string_i = input_pauli[ii][0]
            string_j = input_pauli[jj][0]
            for i in range(string_len):
                if string_i[i] == string_j[i]:
                    if string_i[i] == 'X' or string_i[i] == 'Y':
                        count = count + 1
            res[ii][jj] = XY_count[ii] + XY_count[jj] - count * 2
    return res

############################### Markov Transition Matrices Generation ###############################

def get_markov_0(input_pauli):
    '''
    This function calculates the transition matrix P_qd used in a simplified Markov model.

    The matrix is constructed by duplicating the stationary distribution vector `pro`
    (which represents the normalized absolute values of Hamiltonian coefficients)
    across all rows. The size of the resulting matrix is L x L, where L is the number
    of Pauli strings (i.e., the number of Hamiltonian terms).

    This models a setting where each Hamiltonian term is sampled independently from
    the same fixed distribution at each step.
    '''
    string_num = len(input_pauli)
    pro = []
    lamda = 0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)
    return [copy.deepcopy(pro) for iii in range(string_num)]

def get_markov_1(input_pauli):
    '''
    This function calculates the transition matrix P_gc based on Algorithm 2 from the MarQSim.
    It models a Markov chain over Pauli strings using a minimum-cost flow problem (MCFP) to determine
    optimal transitions between Hamiltonians.

    Key concepts:
        - Each Pauli string is treated as a node in a flow network.
        - Flow represents the probability of transitioning from one Pauli term to another.
        - The cost (or weight) of transitions encodes the number of CNOT and single-qubit gates required.
        - The goal is to minimize the total quantum gate cost while preserving the overall sampling distribution.

    Returns:
        A normalized transition matrix res where res[i][j] is the probability of transitioning
        from Pauli string i to j under the optimized gate cost model.
    '''
    string_num = len(input_pauli)
    pro = []
    lamda = 0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)

    if isinstance(pro[0], list):
        pro = copy.deepcopy(pro[0])

    string_num = len(input_pauli)
    sum_hj_10 = 0
    # Calculate the total flow up to a scaling.
    for ii in range(string_num):
        sum_hj_10 += int(abs(input_pauli[ii][1]) * 10000 + 0.5)
    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)
    single_q_matrix = get_single_q_matrix(input_pauli=input_pauli)

    G = nx.DiGraph()
    # Source point T and sink point T.
    G.add_node('s', demand=-int(sum_hj_10))
    G.add_node('t', demand=int(sum_hj_10))
    # Construct the two sides of the graph with all weight zero.
    for ii in range(string_num):
        G.add_edges_from([('s', (ii, 'b'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
        G.add_edges_from([((ii, 'c'), 't', {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])

    # Construct the middle of the graph.
    for ii in range(string_num):
        for jj in range(string_num):
            if ii == jj:
                continue
            else:
                # G.add_edges_from([((ii, 'b'), (jj, 'c'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                #                                           "weight": int(CNOT_matrix[ii][jj])})])
                G.add_edges_from([((ii, 'b'), (jj, 'c'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                                                          "weight": int(CNOT_matrix[ii][jj]) + int(single_q_matrix[ii][jj])})])
    # Find the minimum cost flow in a directed graph
    # flowCost: This variable will store the cost of the minimum cost flow found in the graph G.
    # flowDict: This variable will store a dictionary representing the flow values on each edge of the graph after the minimum cost flow has been computed. 
    flowCost, flowDict = nx.network_simplex(G)

    flow_matrix = [[0.0 for i in range(string_num)] for i in range(string_num)]
    for head, info in flowDict.items():
        # Ignore the informaiton of flow that start with s, t, (ii, 'c'). 
        if isinstance(head, tuple) == False:
            continue
        if head[1] != 'b':
            continue
        # Head info stands for the flow comes from which 'b' node, 
        # the tail info stands for the flow comes to which 'c' node.
        for tail, flow in info.items():
            flow_matrix[head[0]][tail[0]] = flow
    res = [[0.0 for i in range(string_num)] for i in range(string_num)]

    # normalization
    for ii in range(string_num):
        for jj in range(string_num):
            res[ii][jj] = float(flow_matrix[ii][jj]) / float(int(abs(input_pauli[ii][1]) * 10000 + 0.5))

    return res

def get_markov_2(input_pauli):
    '''
    This function computes a stochastic variant of the optimized transition matrix for Pauli strings.

    It follows a similar setup to `get_markov_1` but introduces randomness into the transition cost
    using a binomial distribution. The idea is to inject noise or variability into the gate-cost model,
    which may be useful for simulating robustness or exploring a probabilistic optimization landscape.

    Specifically:
        - A base CNOT gate cost matrix is computed as in get_markov_1.
        - A random binary matrix (Bernoulli trial with p=0.5) is generated and added to the cost.
        - The resulting costs influence the minimum cost flow (MCF) optimization.

    Returns:
        A normalized transition probability matrix `res` where res[i][j] is the likelihood of
        transitioning from Pauli string i to j under the stochastic cost model.

    By repeating this process multiple times and averaging the outputs (P_gcs), one will obtain P_rp.
    '''
    string_num = len(input_pauli)
    pro = []
    lamda = 0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)

    if isinstance(pro[0], list):
        pro = copy.deepcopy(pro[0])

    string_num = len(input_pauli)
    sum_hj_10 = 0
    for ii in range(string_num):
        sum_hj_10 += int(abs(input_pauli[ii][1]) * 10000 + 0.5)
    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)

    CNOT_matrix_ = np.random.binomial(n=1, p=0.5, size=[string_num, string_num])

    G = nx.DiGraph()
    G.add_node('s', demand=-int(sum_hj_10))
    G.add_node('t', demand=int(sum_hj_10))
    for ii in range(string_num):
        G.add_edges_from([('s', (ii, 'b'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
        G.add_edges_from([((ii, 'c'), 't', {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
    for ii in range(string_num):
        for jj in range(string_num):
            if ii == jj:
                continue
            else:
                G.add_edges_from([((ii, 'b'), (jj, 'c'),
                                   {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                                    "weight": int(CNOT_matrix[ii][jj]) * 100 + int(CNOT_matrix_[ii][jj] * 100)})])
    flowCost, flowDict = nx.network_simplex(G)

    flow_matrix = [[0.0 for i in range(string_num)] for i in range(string_num)]
    for head, info in flowDict.items():
        if isinstance(head, tuple) == False:
            continue
        if head[1] != 'b':
            continue
        for tail, flow in info.items():
            flow_matrix[head[0]][tail[0]] = flow
    res = [[0.0 for i in range(string_num)] for i in range(string_num)]

    for ii in range(string_num):
        for jj in range(string_num):
            res[ii][jj] = float(flow_matrix[ii][jj]) / float(int(abs(input_pauli[ii][1]) * 10000 + 0.5))

    return res

############################### Hamiltonian Evolution Compilation ###############################

def standard_compiler(input_pauli, t, input_x):
    '''
    Evolves the input quantum state `input_x` under the time evolution operator e^{iHt},
    where H is the full Hamiltonian constructed from the weighted Pauli terms in `input_pauli`.
    '''
    hamiltonian = []
    for idx, string in enumerate(input_pauli):
        get_hamilton = get_hamiltonian(string)
        if idx == 0:
            hamiltonian = get_hamilton
        elif idx > 0:
            hamiltonian = hamiltonian + get_hamilton
    op = hamiltonian * (0.0 + 1.0j) * t
    a = torch.matrix_exp(op)
    res = torch.matmul(a, input_x)
    return res

def standard_compiler_pro(input_pauli, t, input_x, N):
    '''
    Applies Trotterized time evolution to the input quantum state `input_x` under the Hamiltonian H,
    using the approximation: (e^{iHt/N})^N ≈ e^{iHt}. The output is the evolved quantum state after applying (e^{iHt/N}) to `input_x`.
    '''
    hamiltonian = []
    for idx, string in enumerate(input_pauli):
        get_hamilton = get_hamiltonian(string)
        if idx == 0:
            hamiltonian = get_hamilton
        elif idx > 0:
            hamiltonian = hamiltonian + get_hamilton
    hamiltonian = hamiltonian / float(N)
    op = torch.matrix_exp(hamiltonian * (0.0 + 1.0j) * t)
    res = copy.deepcopy(input_x)
    for i in range(N):
        res = torch.matmul(op, res)
    return res

def random_compiler_1(
        input_pauli,
        epsilon: float,
        t: float,
        CNOT_matrix=None,
        single_q_matrix=None,
        P_mix=None,
        pro=None
):
    '''
    Compiler for MarQSim based on Algorithm 1 from the paper.

    This function simulates quantum time evolution under a Hamiltonian using randomized
    circuit sampling derived from a Markov transition matrix `P_mix`.

    It computes the approximate unitary e^{iHt} via sequential sampling of Pauli terms and
    returns:
        - The final evolved state,
        - Total CNOT gate count,
        - Total single-qubit gate count,
        - List of sampled Hamiltonian indices.

    Parameters:
        input_pauli (list): Pauli terms as [Pauli_string, coefficient] pairs.
        epsilon (float): Precision parameter for controlling Trotter steps.
        t (float): Total simulation time.
        CNOT_matrix (2D list): CNOT transition cost between Pauli strings.
        single_q_matrix (2D list): Single-qubit transition cost between Pauli strings.
        P_mix (2D list): Markov transition matrix (e.g., from `get_markov_1` or `get_markov_2`).
        pro (list): Stationary distribution over the Pauli strings (same size as input_pauli).

    Returns:
        res (torch.Tensor): The resulting quantum state after compilation.
        CNOT_num (int): Total number of CNOT gates used in the compiled circuit.
        single_q_num (int): Total number of single-qubit gates used.
        sample_list (list): Sequence of sampled Hamiltonian indices used for the circuit.
    '''
    string_num = len(input_pauli)
    lamda = 0.0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    N = 2.0 * lamda * lamda * t * t / epsilon
    N = int(N + 1)

    pro = np.array(pro)

    res = None
    
    curr_sample = 0
    CNOT_num = 0
    single_q_num = 0
    CNOT_start_and_end = 0
    single_q_start_and_end = 0
    sample_list = []

    for idx in range(N):
        if idx == 0:
            sample = np.random.choice([i for i in range(string_num)], p=pro.ravel())
            hamiltonian = get_hamiltonian([input_pauli[sample][0], 1.0])
            sample_list.append(sample)
            if input_pauli[sample][1] < 0:
                hamiltonian = -hamiltonian
            op = torch.matrix_exp(hamiltonian * (0.0 + 1.0j) * lamda * t / float(N))
            res = copy.deepcopy(op)
            curr_sample = sample

            for op in input_pauli[sample][0]:
                if op != 'I':
                    CNOT_start_and_end += 1
                if op == 'X' or op == 'Y':
                    single_q_start_and_end += 1

        else:
            sample = np.random.choice([i for i in range(string_num)], p=(P_mix[curr_sample]).ravel())
            hamiltonian = get_hamiltonian([input_pauli[sample][0], 1.0])
            sample_list.append(sample)
            if input_pauli[sample][1] < 0:
                hamiltonian = -hamiltonian
            op = torch.matrix_exp(hamiltonian * (0.0 + 1.0j) * lamda * t / float(N))
            res = torch.matmul(op, res)
            CNOT_num += CNOT_matrix[curr_sample][sample]
            single_q_num += single_q_matrix[curr_sample][sample]
            curr_sample = sample
            if idx == N - 1:
                for op in input_pauli[sample][0]:
                    if op != 'I':
                        CNOT_start_and_end += 1
                    if op == 'X' or op == 'Y':
                        single_q_start_and_end += 1

    CNOT_num += CNOT_start_and_end
    single_q_num += single_q_start_and_end
    single_q_num += N

    return res, CNOT_num, single_q_num, sample_list


def random_compiler_2(
        input_pauli,
        epsilon: float,
        t: float,
        CNOT_matrix=None,
        single_q_matrix=None,
        P_mix=None,
        pro=None
):
    '''
    A version of the compiler that omits simulation accuracy calculation.
    Useful for analyzing compilation time performance.
    '''

    string_num = len(input_pauli)
    lamda = 0.0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    N = 2.0 * lamda * lamda * t * t / epsilon
    N = int(N + 1)


    pro = np.array(pro)
    
    curr_sample = 0
    CNOT_num = 0
    single_q_num = 0
    CNOT_start_and_end = 0
    single_q_start_and_end = 0
    sample_list = []


    for idx in range(N):
        if idx == 0:
            sample = np.random.choice([i for i in range(string_num)], p=pro.ravel())
            sample_list.append(sample)
            curr_sample = sample

            for op in input_pauli[sample][0]:
                if op != 'I':
                    CNOT_start_and_end += 1
                if op == 'X' or op == 'Y':
                    single_q_start_and_end += 1

        else:
            sample = np.random.choice([i for i in range(string_num)], p=(P_mix[curr_sample]).ravel())
            sample_list.append(sample)
            CNOT_num += CNOT_matrix[curr_sample][sample]
            single_q_num += single_q_matrix[curr_sample][sample]
            curr_sample = sample
            if idx == N - 1:
                for op in input_pauli[sample][0]:
                    if op != 'I':
                        CNOT_start_and_end += 1
                    if op == 'X' or op == 'Y':
                        single_q_start_and_end += 1

    CNOT_num += CNOT_start_and_end
    single_q_num += single_q_start_and_end
    single_q_num += N

    return CNOT_num, single_q_num, sample_list

############################### Helper Function on Data Processing ###############################

# This function is for the sample analysis.
def count_frequency(numbers):
    frequency_dict = {}

    for number in numbers:
        if number in frequency_dict:
            frequency_dict[number] += 1
        else:
            frequency_dict[number] = 1

    sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict

# This function calculates the fidelity using trace
def complex_angle_3(op_1, op_2):
    op_2_ = torch.conj(op_2)
    op_2_ = torch.transpose(op_2_, 0, 1)
    acc = torch.trace(torch.matmul(op_1, op_2_))
    acc = complex(acc)

    acc = (acc.real * acc.real + acc.imag * acc.imag) ** 0.5
    acc = acc / float(op_2.shape[0])

    return acc

############################### MarQSim Experiments ###############################

def operation(
        lam_list: list,
        file:str,
        path: str,
        t: float,
        sampling_time: float,
        epsilon_list: list
):
    '''
    Main experiment runner for MarQSim.

    This function performs simulation experiments by compiling quantum circuits using a
    stochastic Hamiltonian evolution model. It evaluates different mixtures of transition 
    models (QDrift, MarQSim-GC, MarQSim-GC-RP), measuring accuracy and gate cost.

    Parameters:
        lam_list (list of list): List of lambda vectors $[\lambda_0, \lambda_1, \lambda_2]$ that weight the Markov matrices:
                                 QDrift (P_qd), Gate-Cancellation (P_gc), and Gate-Cancellation + Random Perturbation (P_rp).
        file (str): The file name specifies the molecule or ion to be simulated.
        path (str): Path where intermediate information are saved/loaded.
        t (float): Evolution time.
        sampling_time (int): Number of sampling repetitions per $\lambda$ configuration.
        epsilon_list (list): List of epsilon values controlling precision (and thus Trotter step count).

    Returns:
        acc_reses (list of lists): Accuracy results for each λ configuration calculating from trace distance.
        CNOT_num_reses (list of lists): Total CNOT gate counts for each run.
        single_q_num_reses (list of lists): Total single-qubit gate counts for each run.
        samples (list of dicts): Frequency counts of Hamiltonian indices sampled in each run.
    '''

    Kernel_list = get_pauli('Benchmarks' + '//' + '_Pauli_string_' + file[6:])
    
    hamiltonian = []
    for idx, string in enumerate(Kernel_list):
        get_hamilton = get_hamiltonian(string)
        if idx == 0:
            hamiltonian = get_hamilton
        elif idx > 0:
            hamiltonian = hamiltonian + get_hamilton
    op = hamiltonian * (0.0 + 1.0j) * t
    a = torch.matrix_exp(op)
    output_y_0_0 = a

    input_pauli = Kernel_list
    
    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)
    single_q_matrix = get_single_q_matrix(input_pauli=input_pauli)
   
    lamda = 0.0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    pro = []
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)

    P_0 = get_markov_0(input_pauli=input_pauli)
    np.save(path + '/QDrift.npy', P_0)
    P_1 = get_markov_1(input_pauli=input_pauli)
    np.save(path + '/MarQSim-GC.npy', P_1)

    sum_matrix = np.zeros((len(input_pauli),len(input_pauli)))
    for i in range(100):
        sum_matrix += get_markov_2(input_pauli=input_pauli)
    P_2 = sum_matrix/100
    np.save(path + '/MarQSim-GC-RP.npy', P_2)

    P_0 = np.load(path + '/QDrift.npy')
    P_1 = np.load(path + '/MarQSim-GC.npy')
    P_2 = np.load(path + '/MarQSim-GC-RP.npy')

    acc_reses, CNOT_num_reses, single_q_num_reses, samples = [], [], [], []
    for lam in lam_list:
        print("Running lambda configuration:", lam)
        print("Total samples:", sampling_time)
        print("Sampling progress:")
        P_mix = lam[0] * P_0 + lam[1] * P_1 + lam[2] * P_2
        acc_res, CNOT_num_res, single_q_num_res, sample = [], [], [], []
        for idx in tqdm(range(sampling_time)):
            for epsilon in epsilon_list:
                output_y_1, CNOT_num, single_q_num, sample_list = random_compiler_1(
                    input_pauli=copy.deepcopy(Kernel_list), epsilon=epsilon, t=t,
                    CNOT_matrix=CNOT_matrix, single_q_matrix=single_q_matrix, P_mix=P_mix, pro=pro
                )
                output_y_0_0 = output_y_0_0.cpu()
                output_y_1 = output_y_1.cpu()
                acc_0 = complex_angle_3(op_1=output_y_0_0, op_2=output_y_1)
                acc_res.append(acc_0)
                CNOT_num_res.append(CNOT_num)
                single_q_num_res.append(single_q_num)
                sample += sample_list
        acc_reses.append(acc_res)
        CNOT_num_reses.append(CNOT_num_res)
        single_q_num_reses.append(single_q_num_res)
        samples.append(count_frequency(sample))

    return acc_reses, CNOT_num_reses, single_q_num_reses, samples

############################### Date processing ###############################

def total_data(CNOT_numses, single_q_numses, errors, t, lam):
    N_list = [int(2.0 * lam * lam * t * t / error)+1 for error in errors]
    # print(N_list)
    N_num = N_list
    for i in range(19):
        N_num = N_num + N_list
    N_numses = []
    for i in range(3):
        N_numses.append(N_num)
    gate_numses = []
    for i in range(len(CNOT_numses)):
        gate = [x + y for x, y in zip(CNOT_numses[i], single_q_numses[i])]
        gate = [x + y for x, y in zip(gate, N_numses[i])]
        gate_numses.append(gate)
    return gate_numses

def drop_min_elements(nums):
    # Check if the list has at least 2 elements
    if len(nums) < 2:
        return nums  # Nothing to drop
    
    # Sort the list in ascending order
    sorted_nums = sorted(nums)
    
    # Remove the first two elements from the sorted list
    sorted_nums = sorted_nums[2:]
    
    # Create a new list with remaining elements
    result = [num for num in nums if num in sorted_nums]
    
    return result

# Define the model function a + b * e^(c * x)
def model_function(x, a, b, c):
    return a + np.exp(b*x+c)

def read_csv(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        csv_a = csv.DictReader(f)
        sd = []
        sb = []
        ob = []

        for row in csv_a:
            sd.append(float(row["Spectra Distribution"]))
            sb.append(float(row["Spectra Boundary"]))
            ob.append(float(row["Original Boundary"]))

        return {
            "Spectra Distribution": sd,
            "Spectra Boundary": sb,
            "Original Boundary": ob,
        }

def spectra_operation(
        file: str,
        path: str,
):
    
    '''
    Performs the data processing operation required for matrix spectrum analysis.
    '''

    input_pauli = get_pauli('Benchmarks' + '//' + '_Pauli_string_' + file[6:])
   
    lamda = 0.0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    pro = []
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)


    # they are pre calculated
    # by using the commend: P_i = get_markov_i(input_pauli=input_pauli), i = 0, 1, 2
    P_0 = get_markov_0(input_pauli=input_pauli)
    np.save(path + '//' + 'QDrift.npy', P_0)
    
    P_1 = get_markov_1(input_pauli=input_pauli)
    np.save(path + '//' + 'MarQSim-GC.npy', P_1)
    

    sum_matrix = np.zeros((len(input_pauli),len(input_pauli)))
    for i in range(100):
        sum_matrix += get_markov_2(input_pauli=input_pauli)
    P_2 = sum_matrix/100
    np.save(path + '//' + 'MarQSim-GC-RP.npy', P_2)
    # 100

    P_0 = np.load(path + '//' + 'QDrift.npy')
    P_1 = np.load(path + '//' + 'MarQSim-GC.npy')
    P_2 = np.load(path + '//' + 'MarQSim-GC-RP.npy')

    P1 = 0.4*P_0+0.6*P_1
    P1_prime = 0.4*P_0+0.3*P_1+0.3*P_2

    P2 = 0.2*P_0+0.8*P_1
    P2_prime = 0.2*P_0+0.4*P_1+0.4*P_2

    eval_1 = np.sort(np.abs(np.linalg.eigvals(P1)))[::-1]
    eval_1_prime = np.sort(np.abs(np.linalg.eigvals(P1_prime)))[::-1]
    spectra_a_dict = {}
    spectra_a_dict["Spectra Distribution"] = eval_1_prime
    spectra_a_dict["Spectra Boundary"] = eval_1_prime
    spectra_a_dict["Original Boundary"] = eval_1
    df = pd.DataFrame(spectra_a_dict)
    df.to_csv(path + '//' + "eigenspectra_a.csv", index=False)

    eval_2 = np.sort(np.abs(np.linalg.eigvals(P2)))[::-1]
    eval_2_prime = np.sort(np.abs(np.linalg.eigvals(P2_prime)))[::-1]
    spectra_b_dict = {}
    spectra_b_dict["Spectra Distribution"] = eval_2_prime
    spectra_b_dict["Spectra Boundary"] = eval_2_prime
    spectra_b_dict["Original Boundary"] = eval_2
    df = pd.DataFrame(spectra_b_dict)
    df.to_csv(path + '//' + "eigenspectra_b.csv", index=False)

    data_a = read_csv(path + '//' + "eigenspectra_a.csv")
    data_b = read_csv(path + '//' + "eigenspectra_b.csv")

    x = range(1, len(data_a["Spectra Boundary"]) + 1)
    xticks = range(1, len(data_a["Spectra Boundary"]) + 1, 10)

    import matplotlib.pyplot as plt

    plt.rc("font", size=20)
    plt.rc("mathtext", fontset="cm")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(0.06, 0.16, 0.84, 0.93, wspace=0.55)

    plt.subplot(1, 2, 1)
    plt.bar(x, data_a["Spectra Distribution"], label="Spectrum Distribution ($\\mathbf{P_1}'$)")
    plt.plot(
        x, data_a["Spectra Boundary"], color="darkorange", linewidth=5, label="Spectrum of $\\mathbf{P_1}'=0.4\\mathbf{P_{qd}}+0.3\\mathbf{P_{gc}}$\n$+0.3\\mathbf{P_{rp}}$"
    )
    plt.plot(
        x,
        data_a["Original Boundary"],
        color="darkgreen",
        linewidth=4,
        label="Spectrum of $\\mathbf{P}_{1}=0.4\\mathbf{P_{qd}}+0.6\\mathbf{P_{gc}}$",
    )
    plt.title("(a)")
    plt.ylabel("Spectrum Value")
    plt.grid(True, axis="both")
    plt.xticks(xticks, [])
    plt.xlim(1, len(data_a["Spectra Boundary"]))
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [1,0,2]

    #add legend to plot
    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="upper right", bbox_to_anchor=(1.43, 1.02), framealpha=1, edgecolor="black")
    legend.get_frame().set_edgecolor("black")
    plt.xlabel("$\\lambda_i$", fontsize=28)

    plt.subplot(1, 2, 2)
    plt.bar(x, data_b["Spectra Distribution"], label="Spectrum Distribution ($\\mathbf{P_2}'$)")
    plt.plot(
        x, data_b["Spectra Boundary"], color="orange", linewidth=5, label="Spectrum of $\\mathbf{P_2}'=0.2\\mathbf{P_{qd}}+0.4\\mathbf{P_{gc}}$\n$+0.4\\mathbf{P_{rp}}$"
    )
    plt.plot(
        x,
        data_b["Original Boundary"],
        color="darkgreen",
        linewidth=4,
        label="Spectrum of $\\mathbf{P}_{2}=0.2\\mathbf{P_{qd}}+0.8\\mathbf{P_{gc}}$",
    )
    plt.title("(b)")
    plt.grid(True, axis="both")
    plt.xticks(xticks, [])
    plt.xlim(1, len(data_a["Spectra Boundary"]))
    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="upper right", bbox_to_anchor=(1.5, 1.02), framealpha=1)
    legend.get_frame().set_edgecolor("black")
    plt.xlabel("$\\lambda_i$", fontsize=28)
    
    plt.savefig(path + '//' + "Expspectra.png")

    return 0


def compilation_time_operation(
        lam_list: list,
        file:str,
        path: str,
        t: float,
        sampling_time: float,
        epsilon_list: list
):
    
    '''
    Performs the data processing operation required for compilation time analysis.
    '''

    input_pauli = get_pauli('Benchmarks' + '//' + '_Pauli_string_' + file[6:])
    
    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)
    single_q_matrix = get_single_q_matrix(input_pauli=input_pauli)
    
    lamda = 0.0
    for string in input_pauli:
        lamda += max(string[1], -string[1])
    pro = []
    for string in input_pauli:
        p = string[1] / lamda
        p = max(-p, p)
        pro.append(p)

    time_t = []
    
    # print("here")
    t0_s = time.time()
    P_0 = get_markov_0(input_pauli=input_pauli)
    t0_e = time.time()
    t0 = t0_e-t0_s
    time_t.append(t0)
    np.save(path + '/QDrift.npy', P_0)

    # print("here")
    t1_s = time.time()
    P_1 = get_markov_1(input_pauli=input_pauli)
    t1_e = time.time()
    t1 = t1_e-t1_s
    time_t.append(t1)
    np.save(path + '/MarQSim-GC.npy', P_1)

    # print("here")
    t2_s = time.time()
    sum_matrix = np.zeros((len(input_pauli),len(input_pauli)))
    for i in range(1):
        sum_matrix += get_markov_2(input_pauli=input_pauli)
    P_2 = sum_matrix/1
    t2_e = time.time()
    t2 = t2_e-t2_s
    time_t.append(t2)
    np.save(path + '/MarQSim-GC-RP.npy', P_2)

    P_0 = np.load(path + '/QDrift.npy')
    P_1 = np.load(path + '/MarQSim-GC.npy')
    P_2 = np.load(path + '/MarQSim-GC-RP.npy')
    
    CNOT_num_reses, single_q_num_reses, samples, time_circuit = [], [], [], []
    for lam in lam_list:
        P_mix = lam[0] * P_0 + lam[1] * P_1 + lam[2] * P_2
        CNOT_num_res, single_q_num_res = [], []
        for idx in range(sampling_time):
            for epsilon in epsilon_list:
                t_s = time.time()
                CNOT_num, single_q_num, sample_list = random_compiler_2(
                    input_pauli=copy.deepcopy(input_pauli), epsilon=epsilon, t=t,
                    CNOT_matrix=CNOT_matrix, single_q_matrix=single_q_matrix, P_mix=P_mix, pro=pro
                )
                t_e = time.time()
                t_c = t_e-t_s
                time_circuit.append(t_c)
                CNOT_num_res.append(CNOT_num)
                single_q_num_res.append(single_q_num)

        CNOT_num_reses.append(CNOT_num_res)
        single_q_num_reses.append(single_q_num_res)
        samples.append(sample_list)

    return CNOT_num_reses, single_q_num_reses, samples, time_t, time_circuit