'''
    To process the incrementally gathered data in "result.csv", run the following command.
    This will generate figures in each corresponding folder,
    and the reduction rates will be saved in a file named "reduction.txt".

    "python3 data.py --file=Pauli_Na+ --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=10.456",
    "python3 data.py --file=Pauli_Cl- --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=11.13",
    "python3 data.py --file=Pauli_Ar --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=14.61",
    "python3 data.py --file=Pauli_SYK1 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=104.14",
    "python3 data.py --file=Pauli_SYK2 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=108.41",
    "python3 data.py --file=Pauli_OH- --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=18.58",
    "python3 data.py --file=Pauli_HF --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=24.43",
    "python3 data.py --file=Pauli_LiH_f --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=8.89",
    "python3 data.py --file=Pauli_BeH2_f --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=21.49",
    "python3 data.py --file=Pauli_LiH_unf --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=12.34",
    "python3 data.py --file=Pauli_H2O --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=27.16",
    "python3 data.py --file=Pauli_BeH2_unf --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --h_sum=21.49"
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import csv
import sys

file = sys.stdout

import argparse

parser = argparse.ArgumentParser(description="MarQSim Compiler Experiment Settings")
# Path to the Pauli string input file 
parser.add_argument('--file', help='Path to the Pauli string input file.')
# List of epsilon values 
# Should be a string like "0.1,0.01" and will be parsed into a list of floats
parser.add_argument('--epsilon_list_1', help='Comma-separated list of epsilon values.')
# Evolution time `t` in the simulation
parser.add_argument('--execute_time', type=float, help='Total evolution time t.')
# Precomputed or supplied sum of absolute Hamiltonian coefficients
parser.add_argument('--h_sum', type=float, help='Sum of absolute Hamiltonian weights.')

args = parser.parse_args()


epsilon_list_1 = args.epsilon_list_1
epsilon_list_1 = epsilon_list_1.split(',')
epsilon_list = []
for epsilon_str in epsilon_list_1:
    epsilon_list.append(float(epsilon_str))
execute_time = args.execute_time
h_sum = args.h_sum

file_name = args.file + '//' + 'result' + '.csv'

with open(file_name, "r", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)  # first row is header
    rows = list(reader)

columns = list(zip(*rows))  
columns = [[float(x) for x in col] for col in columns]

acc_reses = []
CNOT_numses = []
single_q_numses = []

for i, header in enumerate(headers):
    if header.startswith("acc"):
        acc_reses.append(columns[i])
    elif header.startswith("CNOT"):
        CNOT_numses.append(columns[i])
    elif header.startswith("single_q"):
        single_q_numses.append(columns[i])


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

total_numses = total_data(CNOT_numses, single_q_numses, epsilon_list, execute_time, h_sum)
# CNOT_numses = single_q_numses


# clustering
acc_clusters_1 = [drop_min_elements(acc_reses[0][i::7]) for i in range(7)]
acc_final_1 = np.array([sum(acc_clusters_1[i])/len(acc_clusters_1[i]) for i in range(len(acc_clusters_1))])
acc_std_1 = [np.std(acc_clusters_1[i]) for i in range(len(acc_clusters_1))]
CNOT_clusters_1 = np.array([sum(CNOT_numses[0][i::7])/len(CNOT_numses[0][i::7]) for i in range(7)])
total_clusters_1 = np.array([sum(total_numses[0][i::7])/len(total_numses[0][i::7]) for i in range(7)])
single_clusters_1 = np.array([sum(single_q_numses[0][i::7])/len(single_q_numses[0][i::7]) for i in range(7)])
acc_clusters_2 = [drop_min_elements(acc_reses[1][i::7]) for i in range(7)]
acc_final_2 = np.array([sum(acc_clusters_2[i])/len(acc_clusters_2[i]) for i in range(len(acc_clusters_2))])
acc_std_2 = [np.std(acc_clusters_2[i]) for i in range(len(acc_clusters_2))]
CNOT_clusters_2 = np.array([sum(CNOT_numses[1][i::7])/len(CNOT_numses[1][i::7]) for i in range(7)])
total_clusters_2 = np.array([sum(total_numses[1][i::7])/len(total_numses[1][i::7]) for i in range(7)])
single_clusters_2 = np.array([sum(single_q_numses[1][i::7])/len(single_q_numses[1][i::7]) for i in range(7)])
acc_clusters_3 = [drop_min_elements(acc_reses[2][i::7]) for i in range(7)]
acc_final_3 = np.array([sum(acc_clusters_3[i])/len(acc_clusters_3[i]) for i in range(len(acc_clusters_3))])
acc_std_3 = [np.std(acc_clusters_3[i]) for i in range(len(acc_clusters_3))]
CNOT_clusters_3 = np.array([sum(CNOT_numses[2][i::7])/len(CNOT_numses[2][i::7]) for i in range(7)])
total_clusters_3 = np.array([sum(total_numses[2][i::7])/len(total_numses[2][i::7]) for i in range(7)])
single_clusters_3 = np.array([sum(single_q_numses[2][i::7])/len(single_q_numses[2][i::7]) for i in range(7)])

file_name = args.file + '//' + 'reduction' + '.txt'
output_file = open(file=file_name, mode="a", encoding='utf-8')
x_list = np.array([0.992, 0.9925, 0.993, 0.9935, 0.994])
# CNOT gate reduction
params1, _ = curve_fit(model_function, acc_final_1, CNOT_clusters_1)
a_fit1, b_fit1, c_fit1 = params1
params2, _ = curve_fit(model_function, acc_final_2, CNOT_clusters_2)
a_fit2, b_fit2, c_fit2 = params2
params3, _ = curve_fit(model_function, acc_final_3, CNOT_clusters_3)
a_fit3, b_fit3, c_fit3 = params3
y_fit_CNOT1 = model_function(x_list, a_fit1, b_fit1, c_fit1)
y_fit_CNOT2 = model_function(x_list, a_fit2, b_fit2, c_fit2)
y_fit_CNOT3 = model_function(x_list, a_fit3, b_fit3, c_fit3)
reduce12 = (sum(y_fit_CNOT1)-sum(y_fit_CNOT2))/sum(y_fit_CNOT1)
reduce13 = (sum(y_fit_CNOT1)-sum(y_fit_CNOT3))/sum(y_fit_CNOT1)
print("MarQSim-GC CNOT reduction:", file=output_file)
print("{:.3g}%".format(reduce12 * 100), file=output_file)
print("MarQSim-GC-RP CNOT reduction:", file=output_file)
print("{:.3g}%".format(reduce13 * 100), file=output_file)

# total gate reduction
params1, _ = curve_fit(model_function, acc_final_1, total_clusters_1)
a_fit1, b_fit1, c_fit1 = params1
params2, _ = curve_fit(model_function, acc_final_2, total_clusters_2)
a_fit2, b_fit2, c_fit2 = params2
params3, _ = curve_fit(model_function, acc_final_3, total_clusters_3)
a_fit3, b_fit3, c_fit3 = params3
y_fit_total1 = model_function(x_list, a_fit1, b_fit1, c_fit1)
y_fit_total2 = model_function(x_list, a_fit2, b_fit2, c_fit2)
y_fit_total3 = model_function(x_list, a_fit3, b_fit3, c_fit3)
reduce12 = (sum(y_fit_total1)-sum(y_fit_total2))/sum(y_fit_total1)
reduce13 = (sum(y_fit_total1)-sum(y_fit_total3))/sum(y_fit_total1)
print("MarQSim-GC total gate reduction:", file=output_file)
print("{:.3g}%".format(reduce12 * 100), file=output_file)
print("MarQSim-GC-RP total gate reduction:", file=output_file)
print("{:.3g}%".format(reduce13 * 100), file=output_file)

# one qubit gate reduction
params1, _ = curve_fit(model_function, acc_final_1, single_clusters_1)
a_fit1, b_fit1, c_fit1 = params1
params2, _ = curve_fit(model_function, acc_final_2, single_clusters_2)
a_fit2, b_fit2, c_fit2 = params2
params3, _ = curve_fit(model_function, acc_final_3, single_clusters_3)
a_fit3, b_fit3, c_fit3 = params3
y_fit_single1 = model_function(x_list, a_fit1, b_fit1, c_fit1)
y_fit_single2 = model_function(x_list, a_fit2, b_fit2, c_fit2)
y_fit_single3 = model_function(x_list, a_fit3, b_fit3, c_fit3)
reduce12 = (sum(y_fit_single1)-sum(y_fit_single2))/sum(y_fit_single1)
reduce13 = (sum(y_fit_single1)-sum(y_fit_single3))/sum(y_fit_single1)
print("MarQSim-GC single qubit gate reduction:", file=output_file)
print("{:.3g}%".format(reduce12 * 100), file=output_file)
print("MarQSim-GC-RP single reduction:", file=output_file)
print("{:.3g}%".format(reduce13 * 100), file=output_file)

# calculate the standard deviation reduction
reduce_std = (sum(acc_std_2)-sum(acc_std_3))/sum(acc_std_2)
print("MarQSim-GC-RP standard deviation reduction:", file=output_file)
print("{:.3g}%".format(reduce_std * 100), file=output_file)

# Figure plot
params1, _ = curve_fit(model_function, acc_final_1, CNOT_clusters_1)
a_fit1, b_fit1, c_fit1 = params1
params2, _ = curve_fit(model_function, acc_final_2, CNOT_clusters_2)
a_fit2, b_fit2, c_fit2 = params2
params3, _ = curve_fit(model_function, acc_final_3, CNOT_clusters_3)
a_fit3, b_fit3, c_fit3 = params3
y_fit1 = model_function(acc_final_1, a_fit1, b_fit1, c_fit1)
y_fit2 = model_function(acc_final_2, a_fit2, b_fit2, c_fit2)
y_fit3 = model_function(acc_final_3, a_fit3, b_fit3, c_fit3)

plt.figure(figsize=(8, 8)) 
plt.errorbar(acc_final_1, y_fit1, xerr=acc_std_1, fmt='-x', markersize=16, capsize=7, label='Baseline', color='b', linewidth=2)
plt.errorbar(acc_final_2, y_fit2, xerr=acc_std_2, fmt='-^', markersize=16, capsize=7, label='MarQSim-GC', color='g', linewidth=2)
plt.errorbar(acc_final_3, y_fit3, xerr=acc_std_3, fmt='-o', markersize=16, capsize=7, label='MarQSim-GC-RP', color='y', linewidth=2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
# plt.show()

# Customize the plot
plt.xlabel('Accuracy', fontsize=20)
plt.ylabel('CNOT Gate Count', fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.savefig(args.file + '//photo' + '.png')
