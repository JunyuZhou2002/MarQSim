import subprocess

# List of full Python commands to execute in order
commands = [
    "python3 randomcompiler.py --file=Pauli_10_100 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_20_100 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_30_100 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_10_500 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_20_500 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_30_500 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_10_1000 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_20_1000 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
    "python3 randomcompiler.py --file=Pauli_30_1000 --num=1 --list1=1.0,0.0,0.0,0.4,0.6,0.0,0.4,0.3,0.3 --epsilon_list_1=0.05 --excute_time=0.785 --sampling_time=1",
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