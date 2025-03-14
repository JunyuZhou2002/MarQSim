import subprocess

# List of full Python commands to execute in order
commands = [
    "python3 randomcompiler.py --file=Pauli_Na+ --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=10.456",
    "python3 randomcompiler.py --file=Pauli_Cl- --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=11.13",
    "python3 randomcompiler.py --file=Pauli_Ar --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=14.61",
    "python3 randomcompiler.py --file=Pauli_SYK1 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=104.14",
    "python3 randomcompiler.py --file=Pauli_SYK2 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=108.41",
    "python3 randomcompiler.py --file=Pauli_OH- --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=18.58",
    "python3 randomcompiler.py --file=Pauli_HF --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=24.43",
    "python3 randomcompiler.py --file=Pauli_LiH_f --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=8.89",
    "python3 randomcompiler.py --file=Pauli_BeH2_f --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=21.49",
    "python3 randomcompiler.py --file=Pauli_LiH_unf --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=12.34",
    "python3 randomcompiler.py --file=Pauli_H2O --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=27.16",
    "python3 randomcompiler.py --file=Pauli_BeH2_unf --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=5 --h_sum=21.49"
]

for command in commands:
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the command output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print(f"Error in {command}:\n{result.stderr}")
    
    print(f"Finished: {command}\n{'-'*40}")