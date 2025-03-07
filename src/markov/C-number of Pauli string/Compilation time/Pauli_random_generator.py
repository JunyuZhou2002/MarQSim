import random

# Define the Pauli operators
pauli_operators = ['I', 'X', 'Y', 'Z']

# Function to generate a random Pauli string of a given length
def generate_random_pauli_string(length):
    return ''.join(random.choice(pauli_operators) for i in range(length))

# Function to generate a random coefficient with a random sign
def generate_random_coefficient():
    coefficient = random.uniform(0.1, 1.0)  # Adjust the range as needed
    coefficient = coefficient
    if coefficient < 1e-4:
        coefficient = 1e-4
    sign = random.choice(['+', '-'])
    return f'{sign} {coefficient:.15f}'  # Format the coefficient

# Specify the output file name
output_file = '_Pauli_string_20'

# Generate random Pauli strings with coefficients and signs, and save them to the file
with open(output_file, 'w') as file:
    for i in range(10000):
        coefficient = generate_random_coefficient()
        pauli_string = generate_random_pauli_string(30)
        file.write(f'{coefficient} * {pauli_string}\n')

print(f"Random Pauli strings with coefficients and signs have been generated and saved to '{output_file}'.")