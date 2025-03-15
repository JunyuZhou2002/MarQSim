import subprocess

# List of full Python commands to execute in order
commands = [
    "python randomcompiler.py --file=Pauli_Na+ --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.524 --sampling_time=20 --h_sum=10.456",
    "python randomcompiler.py --file=Pauli_OH- --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.524 --sampling_time=20 --h_sum=18.58",
    "python randomcompiler.py --file=Pauli_Na+ --num=2 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=1.05 --sampling_time=20 --h_sum=10.456",
    "python randomcompiler.py --file=Pauli_OH- --num=2 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=1.05 --sampling_time=20 --h_sum=18.58",
    "python randomcompiler.py --file=Pauli_Na+ --num=3 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=1.57 --sampling_time=20 --h_sum=10.456",
    "python randomcompiler.py --file=Pauli_OH- --num=3 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=1.57 --sampling_time=20 --h_sum=18.58",
    "python randomcompiler.py --file=Pauli_Na+ --num=4 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=2.36 --sampling_time=20 --h_sum=10.456",
    "python randomcompiler.py --file=Pauli_OH- --num=4 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=2.36 --sampling_time=20 --h_sum=18.58"
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