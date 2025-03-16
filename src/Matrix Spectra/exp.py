import subprocess

# List of full Python commands to execute in order
commands = [
    "python3 randomcompiler.py --file=Pauli_Na+ --num=1 --list1=1.0,0.0,0.0,0.8,0.2,0.0,0.4,0.6,0.0,0.2,0.8,0.0 --epsilon_list_1=0.1,0.067,0.05,0.04,0.033,0.0286,0.025 --excute_time=0.785 --sampling_time=20 --h_sum=10.456"]

for command in commands:
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the command output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print(f"Error in {command}:\n{result.stderr}")
    
    print(f"Finished: {command}\n{'-'*40}")