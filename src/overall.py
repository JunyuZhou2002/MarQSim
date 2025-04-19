'''
Experiment Set Up Script. 
It evaluates gate usage (CNOT and single-qubit), fidelity against exact evolution, and analyzes compilation performance across various sampling strategies and parameter settings.

Key functionalities include:
- 1. Support to implement Fig. 14, the Overall Improvement experiment in the paper.
- 2. Support to implement Fig. 17, the Impact of Evolution Time experiment in the paper.

See exp.py on how to execute this file for different experiments.
'''

from marqsim import *
import argparse

parser = argparse.ArgumentParser(description="MarQSim Overall Improvement/Impact of Evolution Time Experiment Settings")
# Path to the experiment folder. 
parser.add_argument('--exp_path', help='Name of the experiment. Used to determine the output folder for saving results.')
# Path to the Pauli string input file 
parser.add_argument('--file', help='Path to the Pauli string input file.')
# Weight configuration for combining the three transition matrices (QDrift, MarQSim-GC, MarQSim-GC-RP)
# Should be a string like "0.2,0.5,0.3" and will be parsed into a list of floats
parser.add_argument('--lam_list_1', help='Comma-separated weights for [P_0, P_1, P_2]. Users can test different configurations; the values will be grouped in sets of three.')
# List of epsilon values 
# Should be a string like "0.1,0.01" and will be parsed into a list of floats
parser.add_argument('--epsilon_list_1', help='Comma-separated list of epsilon values.')
# Evolution time `t` in the simulation
parser.add_argument('--execute_time', type=float, help='Total evolution time t.')
# Number of repetitions (samples) to run for each epsilon value and weight configuration
parser.add_argument('--sampling_time', type=int, help='Number of sampling repetitions.')
# Precomputed or supplied sum of absolute Hamiltonian coefficients
parser.add_argument('--h_sum', type=float, help='Sum of absolute Hamiltonian weights.')

args = parser.parse_args()

lam_list_1 = args.lam_list_1
lam_list_1 = lam_list_1.split(',')
lam_list = []
for idx in range(len(lam_list_1) // 3):
    lam_list.append([float(lam_list_1[idx * 3]), float(lam_list_1[idx * 3 + 1]), float(lam_list_1[idx * 3 + 2])])

epsilon_list_1 = args.epsilon_list_1
epsilon_list_1 = epsilon_list_1.split(',')
epsilon_list = []
for epsilon_str in epsilon_list_1:
    epsilon_list.append(float(epsilon_str))

execute_time = args.execute_time
sampling_time = args.sampling_time
h_sum = args.h_sum

acc_reses, CNOT_numses, single_q_numses = [], [], []
t0 = time.time()

acc_reses, CNOT_numses, single_q_numses, samples = operation(lam_list=lam_list, t=execute_time, file=args.file, path=args.exp_path + '//' + args.file,
                                                    epsilon_list=epsilon_list, sampling_time=sampling_time)
t1 = time.time()

file_name = args.exp_path + '//' + args.file + '//' + 'result' + str(args.execute_time) + '.txt'
output_file = open(file=file_name, mode="a", encoding='utf-8')
print('lam_list\n', lam_list, file=output_file)
print('execute_time\n', execute_time, file=output_file)
print('epsilon_list_1\n', epsilon_list_1, file=output_file)
print('sampling_time\n', sampling_time, file=output_file)
print('acc_reses', file=output_file)
for acc_res in acc_reses:
    print(acc_res, file=output_file)
print('CNOT_numses', file=output_file)
for CNOT_num in CNOT_numses:
    print(CNOT_num, file=output_file)
print('single_q_numses', file=output_file)
for single_q_num in single_q_numses:
    print(single_q_num, file=output_file)


print('Compilation Time', file=output_file)
print(t1 - t0, file=output_file)



total_numses = total_data(CNOT_numses, single_q_numses, epsilon_list, execute_time, h_sum)

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
plt.savefig(args.exp_path + '//' + args.file + '//result' + str(args.execute_time) + '.png')


# commend line
'''
python3 overall.py --exp_path=Overall_Improvement --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=10.456
'''
