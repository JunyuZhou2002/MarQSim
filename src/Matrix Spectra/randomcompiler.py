import copy
import numpy as np
import scipy
import networkx as nx
import torch
from tqdm import tqdm
import random
from scipy.optimize import curve_fit
import pandas as pd
import csv

import sys

file = sys.stdout
'''
source in Pauli\\Pauli_module
'''
import time

t0 = time.time()
device = 'cuda'
torch.set_default_dtype(torch.float64)      
np.random.seed(2012)

# This function pull out the pauli string information contained in the file _Pauli_string_, 
# the output is like [..., ['IIIIIIZI', -0.9493], ...]
def get_pauli(path: str):
    file = open(file=path, mode="r", encoding='utf-8')
    file = file.readlines()
    res = []
    for line in file:
        if len(line) < 2:
            continue
        line = line.split(' ')
        # print(line)
        val = int(float(line[1]) * 10000 + 0.5)
        if val == 0:
            continue
        val = float(val) / 10000
        # print(line)
        # print(line[3][:-1])
        if line[0] == '+':
            res.append([line[3][:-1], val])
        if line[0] == '-':
            res.append([line[3][:-1], -val])
    # print(res)
    # print(len(res))
    return res

# input list like ['IIIIIIZI', -0.9493], and calculate the matrix representation for w_{j}P_{j},
# where w_{j} = list[1] and P_{j} is calculated by the tensor product of pauli string
def get_halmiton(list):
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

# the input_pauli is Kernel_list, the output of get_pauli
# the output matrix stand for: M[i,j]: the number of CNOT we need to transfer the pauli i to pauli j
def get_CNOT_matrix(input_pauli):
    string_num = len(input_pauli)
    # string length stand for the total qubit in the problem
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

# the output matrix stand for: M[i,j]: the number of single quantum gates we need to transfer the pauli i to pauli j
def get_single_q_matrix(input_pauli):
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


# calculate the transition matrix 0, 
# which is constructed by dupilcate the stationary distribution for L times, L stands for the total number of hamiltonian.
def get_markov_0(input_pauli):
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
    # calculate the total flow up to a scaling
    for ii in range(string_num):
        sum_hj_10 += int(abs(input_pauli[ii][1]) * 10000 + 0.5)
    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)
    single_q_matrix = get_single_q_matrix(input_pauli=input_pauli)

    G = nx.DiGraph()
    # start point s and end point t
    G.add_node('s', demand=-int(sum_hj_10))
    G.add_node('t', demand=int(sum_hj_10))
    # construct the teo side of the graph with all weight zero
    for ii in range(string_num):
        G.add_edges_from([('s', (ii, 'b'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
        G.add_edges_from([((ii, 'c'), 't', {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])

    # construct the middle of the graph
    for ii in range(string_num):
        for jj in range(string_num):
            if ii == jj:
                continue
            else:
                # G.add_edges_from([((ii, 'b'), (jj, 'c'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                #                                           "weight": int(CNOT_matrix[ii][jj])})])
                G.add_edges_from([((ii, 'b'), (jj, 'c'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                                                          "weight": int(CNOT_matrix[ii][jj]) + int(single_q_matrix[ii][jj])})])
    # find the minimum cost flow in a directed graph
    # flowCost: This variable will store the cost of the minimum cost flow found in the graph G.
    # flowDict: This variable will store a dictionary representing the flow values on each edge of the graph after the minimum cost flow has been computed. 
    flowCost, flowDict = nx.network_simplex(G)

    flow_matrix = [[0.0 for i in range(string_num)] for i in range(string_num)]
    for head, info in flowDict.items():
        # ignore the informaiton of flow that start with s, t, (ii, 'c') 
        if isinstance(head, tuple) == False:
            continue
        if head[1] != 'b':
            continue
        # head info stands for the flow comes from which 'b' node, 
        # the tail info stands for the flow comes to which 'c' node,
        for tail, flow in info.items():
            flow_matrix[head[0]][tail[0]] = flow
    # print(flow_matrix)
    res = [[0.0 for i in range(string_num)] for i in range(string_num)]

    # normalization
    for ii in range(string_num):
        for jj in range(string_num):
            res[ii][jj] = float(flow_matrix[ii][jj]) / float(int(abs(input_pauli[ii][1]) * 10000 + 0.5))
    # print(np.array(res))
    return res


# give a disturbulent on the CNOT_matrix
def get_markov_2(input_pauli):

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
    # CNOT_matrix_ = np.zeros((string_num, string_num))
    # print(CNOT_matrix_)

    G = nx.DiGraph()
    G.add_node('s', demand=-int(sum_hj_10))
    G.add_node('t', demand=int(sum_hj_10))
    for ii in range(string_num):
        G.add_edges_from([('s', (ii, 'b'), {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
        G.add_edges_from([((ii, 'c'), 't', {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5), "weight": 0.0})])
    # print('NODES\n', G.nodes)
    for ii in range(string_num):
        for jj in range(string_num):
            if ii == jj:
                continue
            else:
                G.add_edges_from([((ii, 'b'), (jj, 'c'),
                                   {"capacity": int(abs(input_pauli[ii][1]) * 10000 + 0.5),
                                    "weight": int(CNOT_matrix[ii][jj]) * 100 + int(CNOT_matrix_[ii][jj] * 100)})])
                # {"capacity": 99.9, "weight": float(CNOT_matrix[ii][jj]) + 0.2*float(CNOT_matrix_[ii][jj])})])
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


# input_x is the original state, this function return the post state of x after the evelution of e^{iHt}
def standard_compiler(input_pauli, t, input_x):
    halmiton = []
    for idx, string in enumerate(input_pauli):
        get_halmito = get_halmiton(string)
        if idx == 0:
            halmiton = get_halmito
        elif idx > 0:
            halmiton = halmiton + get_halmito
    op = halmiton * (0.0 + 1.0j) * t
    a = torch.matrix_exp(op)
    res = torch.matmul(a, input_x)
    return res


# the compiler that use the form e^{iHt/N}, the output is only 1/N of the total hamiltonian 
def standard_compiler_pro(input_pauli, t, input_x, N):
    halmiton = []
    for idx, string in enumerate(input_pauli):
        get_halmito = get_halmiton(string)
        if idx == 0:
            halmiton = get_halmito
        elif idx > 0:
            halmiton = halmiton + get_halmito
    halmiton = halmiton / float(N)
    op = torch.matrix_exp(halmiton * (0.0 + 1.0j) * t)
    res = copy.deepcopy(input_x)
    for i in range(N):
        res = torch.matmul(op, res)
    return res

# this function is for the sample analysis
def count_frequency(numbers):
    frequency_dict = {}

    for number in numbers:
        if number in frequency_dict:
            frequency_dict[number] += 1
        else:
            frequency_dict[number] = 1

    sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict

# this is the compiler for Markov Simulation
# it returns the CNOT gate and single qubit gate account in the compilation
# also return the unique hamiltonian in the sampling
def random_compiler_1(
        input_pauli,
        epsilon: float,
        t: float,
        CNOT_matrix=None,
        single_q_matrix=None,
        P_mix=None,
        pro=None
):
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

    for idx in tqdm(range(N)):
        if idx == 0:
            sample = np.random.choice([i for i in range(string_num)], p=pro.ravel())
            halmiton = get_halmiton([input_pauli[sample][0], 1.0])
            sample_list.append(sample)
            if input_pauli[sample][1] < 0:
                halmiton = -halmiton
            op = torch.matrix_exp(halmiton * (0.0 + 1.0j) * lamda * t / float(N))
            res = copy.deepcopy(op)
            curr_sample = sample

            for op in input_pauli[sample][0]:
                if op != 'I':
                    CNOT_start_and_end += 1
                if op == 'X' or op == 'Y':
                    single_q_start_and_end += 1

        else:
            sample = np.random.choice([i for i in range(string_num)], p=(P_mix[curr_sample]).ravel())
            halmiton = get_halmiton([input_pauli[sample][0], 1.0])
            sample_list.append(sample)
            if input_pauli[sample][1] < 0:
                halmiton = -halmiton
            op = torch.matrix_exp(halmiton * (0.0 + 1.0j) * lamda * t / float(N))
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


# it pull out the element for the last qubit
def sub_1(input_pauli: list):
    res = []
    for gate in input_pauli:
        string_ = gate[0][:-1]
        res.append([string_, gate[1]])
    return res


# using this function to process the row data of CNOT_numses and acc_reses
def data_process(CNOT_numses, acc_reses, step, alpha):
    num = len(CNOT_numses)
    group_CNOT_numses = []
    group_acc_reses = []

    for i in range(num):
        CNOT = CNOT_numses[i]
        acc = acc_reses[i]

        maximum = max(CNOT)
        minimum = min(CNOT)
        k_max = int((maximum + step - 0.00001) // step)
        k_min = int(minimum // step)
        rounded_max = k_max * step
        rounded_min = k_min * step

        group_CNOT = [k*step+step/2 for k in range(k_min, k_max)]
        group_acc_0 = [[] for k in range(k_min, k_max)]

        num_ = len(CNOT)
        for j in range(num_):
            k = CNOT[j]
            k = int(k // step -k_min)
            group_acc_0[k].append(acc[j])


        group_acc = [sum(group_acc_0[k-k_min])/len(group_acc_0[k-k_min]) if len(group_acc_0[k-k_min]) != 0 else 0 for k in range(k_min, k_max)]
 
        count = 0
        for j in range(len(group_acc)):
            if group_acc[j-count] == 0:
                group_acc.pop(j-count)
                group_CNOT.pop(j-count)
                count = count+1

        num_remove = int(len(group_acc) * alpha)
        if num_remove > 0:
            group_acc = group_acc[num_remove:-num_remove]
            group_CNOT = group_CNOT[num_remove:-num_remove]


        group_CNOT_numses.append(group_CNOT)
        group_acc_reses.append(group_acc)

    return group_CNOT_numses, group_acc_reses



def calculate_jsd_between_dicts(dict1, dict2, epsilon=1e-10):
    def js_divergence(p, q):
        # Add epsilon to avoid taking the logarithm of zero
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        
        # Calculate the average KL divergence
        m = 0.5 * (p + q)
        kl_p = np.sum(p * np.log2(p / m))
        kl_q = np.sum(q * np.log2(q / m))
        avg_kl_divergence = 0.5 * (kl_p + kl_q)
        
        # Calculate the Jensen-Shannon Divergence
        js_divergence = np.sqrt(avg_kl_divergence)
        return js_divergence
    
    # Get the set of all unique keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    
    # Convert frequency distributions to probability distributions
    prob_dict1 = {k: dict1.get(k, 0) / sum(dict1.values()) for k in all_keys}
    prob_dict2 = {k: dict2.get(k, 0) / sum(dict2.values()) for k in all_keys}
    
    # Calculate Jensen-Shannon Divergence
    js_distance = js_divergence(np.array(list(prob_dict1.values())), np.array(list(prob_dict2.values())))
    
    return js_distance



import time
from numpy import mat

epsilon_list_1 = [1.0 / float(i) for i in range(10, 50)]


# acculate the fidelity of two matrices
def complex_angle_1(vec_1, vec_2):
    list_1, list_2 = mat(vec_1).tolist()[0], mat(vec_2).tolist()[0]
    len_ = len(list_1)
    acc = 0
    for idx in range(len_):
        acc += list_1[idx].real * list_2[idx].real + list_1[idx].imag * list_2[idx].imag
    return acc


def complex_angle_2(vec_1, vec_2):
    acc = (vec_1 * (vec_2.conjugate())).sum()

    acc = (acc.real * acc.real + acc.imag * acc.imag) ** 0.5

    return acc


# calculate the fidelity using trace
def complex_angle_3(op_1, op_2):
    op_2_ = torch.conj(op_2)
    op_2_ = torch.transpose(op_2_, 0, 1)
    acc = torch.trace(torch.matmul(op_1, op_2_))
    acc = complex(acc)

    acc = (acc.real * acc.real + acc.imag * acc.imag) ** 0.5
    acc = acc / float(op_2.shape[0])

    return acc

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


# the main function of this task
# the return are 
def operation(
        lam_list: list,
        path: str,
        t: float,
        sampling_time: float,
        epsilon_list: list
):
    # epsilon_list=epsilon_list_1
    Kernel_list = get_pauli('_Pauli_string_' + path[6:])



    # this part get the classical hamiltonian
    halmiton = []
    for idx, string in enumerate(Kernel_list):
        get_halmito = get_halmiton(string)
        if idx == 0:
            halmiton = get_halmito
        elif idx > 0:
            halmiton = halmiton + get_halmito
    op = halmiton * (0.0 + 1.0j) * t
    a = torch.matrix_exp(op)
    output_y_0_0 = a

    # print(output_y_0)

    input_pauli = Kernel_list
    

    CNOT_matrix = get_CNOT_matrix(input_pauli=input_pauli)
    # print(CNOT_matrix)
    single_q_matrix = get_single_q_matrix(input_pauli=input_pauli)
    # print(single_q_matrix)
   
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
    np.save('QDrift.npy', P_0)
    
    P_1 = get_markov_1(input_pauli=input_pauli)
    np.save('MarQSim-GC.npy', P_1)
    

    sum_matrix = np.zeros((len(input_pauli),len(input_pauli)))
    for i in range(100):
        sum_matrix += get_markov_2(input_pauli=input_pauli)
    P_2 = sum_matrix/100
    np.save('MarQSim-GC-RP.npy', P_2)
    # 100

    P_0 = np.load('QDrift.npy')
    P_1 = np.load('MarQSim-GC.npy')
    P_2 = np.load('MarQSim-GC-RP.npy')

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
    df.to_csv("eigenspectra_a.csv", index=False)

    eval_2 = np.sort(np.abs(np.linalg.eigvals(P2)))[::-1]
    eval_2_prime = np.sort(np.abs(np.linalg.eigvals(P2_prime)))[::-1]
    spectra_b_dict = {}
    spectra_b_dict["Spectra Distribution"] = eval_2_prime
    spectra_b_dict["Spectra Boundary"] = eval_2_prime
    spectra_b_dict["Original Boundary"] = eval_2
    df = pd.DataFrame(spectra_b_dict)
    df.to_csv("eigenspectra_b.csv", index=False)

    data_a = read_csv("eigenspectra_a.csv")
    data_b = read_csv("eigenspectra_b.csv")

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
    
    plt.savefig("Expspectra.png")

    return 0


import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file')
# this num is not a parameter
parser.add_argument('--num', type=int)
# lsit1 is created for the weight of three transition matrix
parser.add_argument('--list1')
parser.add_argument('--epsilon_list_1')
# t
parser.add_argument('--excute_time', type=float)
# N
parser.add_argument('--sampling_time', type=int)
parser.add_argument('--h_sum', type=float)

args = parser.parse_args()

list1 = args.list1
list1 = list1.split(',')
lam_list = []
for idx in range(len(list1) // 3):
    lam_list.append([float(list1[idx * 3]), float(list1[idx * 3 + 1]), float(list1[idx * 3 + 2])])

epsilon_list_1 = args.epsilon_list_1
epsilon_list_1 = epsilon_list_1.split(',')
epsilon_list = []
for epsilon_str in epsilon_list_1:
    epsilon_list.append(float(epsilon_str))

excute_time = args.excute_time
sampling_time = args.sampling_time
h_sum = args.h_sum

operation(lam_list=lam_list, t=excute_time, path=args.file,
                                                    epsilon_list=epsilon_list, sampling_time=sampling_time)



