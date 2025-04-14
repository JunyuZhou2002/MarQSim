### We provide the command for running experiments and incrementally gathering the data.
### Now you can define the sampling_time argument for each experiment, as long as the total sampling_time adds up to 20
### (with the exception of Pauli_BeH2_unf, which should adds up to 5).
### For each data collection run, make sure to change the random_seed argument. Otherwise, you'll end up collecting the same data multiple times.
### The results will be automatically gathered and saved in "result.csv" under the corresponding molecule/ion folder.
### This file supports incremental appending — existing data will not be overwritten.
'''
    "python3 randomcompiler.py --file=Pauli_Na+ --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=10.456 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_Cl- --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=11.13 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_Ar --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=14.61 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_SYK1 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=104.14 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_SYK2 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=108.41 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_OH- --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=18.58 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_HF --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=24.43 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_LiH_f --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=8.89 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_BeH2_f --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=21.49 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_LiH_unf --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=12.34 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_H2O --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=27.16 --random_seed=2012",
    "python3 randomcompiler.py --file=Pauli_BeH2_unf --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=5 --h_sum=21.49 --random_seed=2012"
'''
### To process the data in "result.csv", run the following command.
### This will generate figures in each corresponding folder as before,
### and the reduction rates will be saved in a file named "reduction.txt".
'''
    "python3 data.py --file=Pauli_Na+ --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=10.456",
    "python3 data.py --file=Pauli_Cl- --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=11.13",
    "python3 data.py --file=Pauli_Ar --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=14.61",
    "python3 data.py --file=Pauli_SYK1 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=104.14",
    "python3 data.py --file=Pauli_SYK2 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=108.41",
    "python3 data.py --file=Pauli_OH- --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=18.58",
    "python3 data.py --file=Pauli_HF --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=24.43",
    "python3 data.py --file=Pauli_LiH_f --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=8.89",
    "python3 data.py --file=Pauli_BeH2_f --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=21.49",
    "python3 data.py --file=Pauli_LiH_unf --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=12.34",
    "python3 data.py --file=Pauli_H2O --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=27.16",
    "python3 data.py --file=Pauli_BeH2_unf --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --h_sum=21.49"
'''