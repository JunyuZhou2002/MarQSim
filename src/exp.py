'''
This file will guide you how to reproduce the experiments show on MarQSim paper.

To reproduce Fig. 14, run: python3 exp.py --experiment=overall
To reproduce Fig. 15, run: python3 exp.py --experiment=varying
To reproduce Fig. 16, run: python3 exp.py --experiment=spectra
To reproduce Fig. 17, run: python3 exp.py --experiment=evol
To reproduce Table. 2, run: python3 exp.py --experiment=time

Note: You can also copy and paste each command string from the experiment list and execute them individually. This way, a process indicator will be displayed as well.

'''

import subprocess
import argparse

parser = argparse.ArgumentParser(description="MarQSim Compiler Experiments")
parser.add_argument('--experiment', help='Name of the experiment. Used to determine which experiment you want to reproduce.')

command_overall = [
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=10.456",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_Cl- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=11.13",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_Ar --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=14.61",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_SYK1 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=104.14",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_SYK2 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=108.41",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=18.58",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_HF --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=24.43",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_LiH_f --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=8.89",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_BeH2_f --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=21.49",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_LiH_unf --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=12.34",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_H2O --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=27.16",
    "python3 overall.py --exp_path=Overall_Improvement --file=Pauli_BeH2_unf --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=5 --h_sum=21.49"
]

'''
    We provide the command for running experiments and incrementally gathering the data.
    You can define the sampling_time argument for each experiment, as long as the total sampling_time adds up to 20
    (with the exception of Pauli_BeH2_unf, which should adds up to 5).
    For each data collection run, make sure to change the random_seed argument. Otherwise, you'll end up collecting the same data multiple times.
    The results will be automatically gathered and saved in "result.csv" under the corresponding molecule/ion folder under "Overall_Parallel".
    This file supports incremental appending — existing data will not be overwritten.
    After finishing the data collection, execute the data.py under "Overall_Parallel" for data processing.

    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=10.456 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_Cl- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=11.13 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_Ar --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=14.61 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_SYK1 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=104.14 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_SYK2 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=108.41 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=18.58 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_HF --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=24.43 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_LiH_f --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=8.89 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_BeH2_f --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=21.49 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_LiH_unf --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=12.34 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_H2O --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=27.16 --random_seed=2012",
    "python3 overall_parallel.py --exp_path=Overall_Parallel --file=Pauli_BeH2_unf --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=5 --h_sum=21.49 --random_seed=2012"

'''

command_varying = [
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=10.456",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_Cl- --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=11.13",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_Ar --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=14.61",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_SYK1 --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=104.14",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_SYK2 --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=108.41",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=18.58",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_HF --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=24.43",
    "python3 varying.py --exp_path=Varying_Combination --file=Pauli_LiH_unf --lam_list_1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.785 --sampling_time=20 --h_sum=12.34"
    ]

command_evol = [
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.524 --sampling_time=20 --h_sum=10.456",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=0.524 --sampling_time=20 --h_sum=18.58",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=1.05 --sampling_time=20 --h_sum=10.456",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=1.05 --sampling_time=20 --h_sum=18.58",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=1.57 --sampling_time=20 --h_sum=10.456",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=1.57 --sampling_time=20 --h_sum=18.58",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_Na+ --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=2.36 --sampling_time=20 --h_sum=10.456",
    "python3 overall.py --exp_path=Evolution_Time_Impact --file=Pauli_OH- --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --execute_time=2.36 --sampling_time=20 --h_sum=18.58"
    ]

command_spectra = [
    "python3 spectra.py --exp_path=Matrix_Spectra --file=Pauli_Na+"
    ]

command_time = [
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_10_100 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_20_100 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_30_100 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_10_500 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_20_500 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_30_500 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_10_1000 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_20_1000 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    "python3 compile_time.py --exp_path=Compilation_Time --file=Pauli_30_1000 --lam_list_1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --execute_time=0.785 --sampling_time=1",
    ]

args = parser.parse_args()
experiment = args.experiment

if experiment == "overall":
    commands = command_overall
elif experiment == "varying":
    commands = command_varying
elif experiment == "evol":
    commands = command_evol
elif experiment == "spectra":
    commands = command_spectra
elif experiment == "time":
    commands = command_time

for command in commands:
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the command output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print(f"Error in {command}:\n{result.stderr}")
    
    print(f"Finished: {command}\n{'-'*40}")