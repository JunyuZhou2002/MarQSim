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
# Precomputed or supplied sum of absolute Hamiltonian coefficients
parser.add_argument('--h_sum', type=float, help='Sum of absolute Hamiltonian weights.')
# Random seed used to collect data
parser.add_argument('--random_seed', type=float, help='Specify different random seed each time when incrementally collecting data.')

args = parser.parse_args()

random_seed = int(args.random_seed)
np.random.seed(random_seed)

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



file_name = args.exp_path + '//' + args.file + '//' + 'result' + '.txt'
output_file = open(file=file_name, mode="a", encoding='utf-8')
print('lam_list\n', lam_list, file=output_file)
print('excute_time\n', execute_time, file=output_file)
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

# Transpose the data to align each list's sub-elements as columns
acc_columns = list(zip(*acc_reses))
cnot_columns = list(zip(*CNOT_numses))
single_q_columns = list(zip(*single_q_numses))

filename = args.exp_path + '//' + args.file + '//' + 'result' + '.csv'
# Create headers only if needed
headers = (
    [f"acc{i+1}" for i in range(len(acc_reses))] +
    [f"CNOT{i+1}" for i in range(len(CNOT_numses))] +
    [f"single_q{i+1}" for i in range(len(single_q_numses))]
)

# Transpose column data into rows
all_columns = acc_reses + CNOT_numses + single_q_numses
rows = list(zip(*all_columns))

# Append mode — keeps all existing data intact
with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
        writer.writerow(headers)
    writer.writerows(rows)


# commend line
'''
python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=10.456 --random_seed=2012
'''
