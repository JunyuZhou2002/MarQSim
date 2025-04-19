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

parser = argparse.ArgumentParser(description="MarQSim Compiler Experiment Settings")
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

CNOT_numses, single_q_numses = [], []

CNOT_numses, single_q_numses, samples, time_t, time_circuit = compilation_time_operation(lam_list=lam_list, t=execute_time, file=args.file, path=args.exp_path + '//' + args.file,
                                                    epsilon_list=epsilon_list, sampling_time=sampling_time)


file_name = args.exp_path + '//' + args.file + '//' + 'result' + '.txt'
output_file = open(file=file_name, mode="a", encoding='utf-8')
print('lam_list\n', lam_list, file=output_file)
print('excute_time\n', execute_time, file=output_file)
print('epsilon_list_1\n', epsilon_list_1, file=output_file)

print('transition compile time', file=output_file)
print('P_qd:', file=output_file)
print(time_t[0], file=output_file)
print('P_gc:', file=output_file)
print(time_t[1], file=output_file)
print('P_rp:', file=output_file)
print(time_t[2], file=output_file)
print('\n', file=output_file)
print('circuit compile time', file=output_file)
print('Baseline:', file=output_file)
print(time_circuit[0], file=output_file)
print('MarQSim-GC:', file=output_file)
print(time_circuit[1], file=output_file)
print('MarQSim-GC-RP:', file=output_file)
print(time_circuit[2], file=output_file)


# commend line
'''
python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_10_100 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1
'''