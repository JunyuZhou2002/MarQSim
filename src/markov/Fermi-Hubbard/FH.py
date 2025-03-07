### This file change the Majorana operator to Hamiltonian

t=1 
U=4

# Function read the majorana operator
# Majorana = get_majorana('FH_8.txt')
def get_majorana(path: str):
    file = open(file=path, mode="r", encoding='utf-8')
    file = file.readlines()
    Majorana = []
    for line in file:
        # print(line[:-1])
        Majorana.append(line[:-1])
    return Majorana



# Function for single Pauli string product
# sign, pauli = single_pauli_product('X', 'Y')
def single_pauli_product(pauli1, pauli2):
    
    if pauli1 == 'I':
        sign = 1
        pauli = pauli2
    elif pauli2 == 'I':
        sign = 1
        pauli = pauli1
    elif pauli1 == pauli2:
        sign = 1
        pauli = 'I'
    elif pauli1 == 'X' and pauli2 == 'Y':
        sign = 1j
        pauli = 'Z'
    elif pauli1 == 'X' and pauli2 == 'Z':
        sign = -1j
        pauli = 'Y'
    elif pauli1 == 'Y' and pauli2 == 'Z':
        sign = 1j
        pauli = 'X'
    elif pauli1 == 'Y' and pauli2 == 'X':
        sign = -1j
        pauli = 'Z'
    elif pauli1 == 'Z' and pauli2 == 'X':
        sign = 1j
        pauli = 'Y'
    elif pauli1 == 'Z' and pauli2 == 'Y':
        sign = -1j
        pauli = 'X'
    else:
        raise ValueError("Invalid Pauli operators: {} and {}".format(pauli1, pauli2))

    return sign, pauli



# Function for multiple pauli string product
# sign, pauli = pauli_string_product('XXXZI', 'YYXXX')
def pauli_string_product(pauli_string1, pauli_string2):
    if len(pauli_string1) != len(pauli_string2):
        raise ValueError("Pauli strings must have the same length")
    sign = 1
    pauli = ''
    for qubit1, qubit2 in zip(pauli_string1, pauli_string2):
        sub_sign, sub_pauli = single_pauli_product(qubit1, qubit2)
        sign *= sub_sign
        pauli += sub_pauli
    return sign, pauli



# function for operator product
# the operator takes form [0.5, 'IIIZIIIX', (-0-0.5j), 'IIIYIIIX']
# operator = operator_product([0.5, 'IIIZIIIX', (-0-0.5j), 'IIIYIIIX'], [0.5, 'IIIZIIIX', (-0-0.5j), 'IIIYIIIX', 11, 'IIIIIIII'])
def operator_product(operator1, operator2):
    
    result_dict = {}

    for i in range(0, len(operator1), 2):
        for j in range(0, len(operator2), 2):
            coefficient1 = operator1[i]
            pauli_string1 = operator1[i + 1]
            coefficient2 = operator2[j]
            pauli_string2 = operator2[j + 1]
            sign, pauli_string = pauli_string_product(pauli_string1, pauli_string2)
            coefficient = coefficient1*coefficient2*sign
            # add it to dictionary
            if pauli_string not in result_dict:
                result_dict[pauli_string] = coefficient
            else:
                result_dict[pauli_string] += coefficient
            
    operator = []
    # Iterate through the dictionary to access both keys and values
    for pauli_string, coefficient in result_dict.items():
        operator.append(coefficient)
        operator.append(pauli_string)
    # print(operator)
    return operator


### Funtion for operator addition
# operator = operator_add([0.5, 'IIIZIIIX', (-0-0.5j), 'IIIYIIIX'], [0.5, 'IIIZIIIX', (-0-0.5j), 'IIIYIIIX', 11, 'IIIIIIII'])
def operator_add(operator1, operator2):
    add_dict = {}
    for i in range(0, len(operator1), 2):
        coefficient = operator1[i]
        pauli_string = operator1[i + 1]
        add_dict[pauli_string] = coefficient
    for i in range(0, len(operator2), 2):
        coefficient = operator2[i]
        pauli_string = operator2[i + 1]
        if pauli_string not in add_dict:
            add_dict[pauli_string] = coefficient
        else:
            add_dict[pauli_string] += coefficient
    operator = []
    for pauli_string, coefficient in add_dict.items():
        if coefficient != 0:
            operator.append(coefficient)
            operator.append(pauli_string)    
    return operator



# Function to calculate the creation and annihilation operator from majorana operator
# creation_up, creation_down, annihilation_up, annihilation_down = majorana_to_qubit_operator(Majorana)
def majorana_to_qubit_operator(Majorana):
    N = int(len(Majorana)/2)
    creation_list = []
    annihilation_list = []
    for i in range(N):
        a_dagger = [1/2, Majorana[2*i], -1j/2, Majorana[2*i+1]]
        a = [1/2, Majorana[2*i], 1j/2, Majorana[2*i+1]]
        creation_list.append(a_dagger)
        annihilation_list.append(a)

    creation_up = []
    creation_down = []
    for idx, value in enumerate(creation_list):
        if idx % 2 == 0:
            creation_up.append(value)
        else:
            creation_down.append(value)

    annihilation_up = []
    annihilation_down = []
    for idx, value in enumerate(annihilation_list):
        if idx % 2 == 0:
            annihilation_up.append(value)
        else:
            annihilation_down.append(value)

    return creation_up, creation_down, annihilation_up, annihilation_down



# Function to calculate first term in Hubbard model
def first_term(creation_up, creation_down, annihilation_up, annihilation_down):
    list = []
    # up
    for i in range(len(creation_up)-1):
        l1 = creation_up[i]
        l2 = annihilation_up[i+1]
        l1l2 = operator_product(l1, l2)
        l3 = creation_up[i+1]
        l4 = annihilation_up[i]
        l3l4 = operator_product(l3, l4)
        l1l2l3l4 = operator_add(l1l2, l3l4)
        list.append(l1l2l3l4)
    # down
    for i in range(len(creation_down)-1):
        l1 = creation_down[i]
        l2 = annihilation_down[i+1]
        l1l2 = operator_product(l1, l2)
        l3 = creation_down[i+1]
        l4 = annihilation_down[i]
        l3l4 = operator_product(l3, l4)
        l1l2l3l4 = operator_add(l1l2, l3l4)
        list.append(l1l2l3l4)
    # add the term in list
    operator = list[0]
    for i in range(len(list)-1):
        operator = operator_add(operator, list[i+1])
    for i in range(0, len(operator), 2):
        operator[i] = -t*operator[i]
    return operator



# Function to calculate second term in Hubabrd model
def second_term(creation_up, creation_down, annihilation_up, annihilation_down):
    list = []
    for i in range(len(creation_up)):
        l1 = creation_up[i]
        l2 = creation_down[i]
        l1l2 = operator_product(l1, l2)
        l3 = annihilation_down[i]
        l1l2l3 = operator_product(l1l2, l3)
        l4 = annihilation_up[i]
        l1l2l3l4 = operator_product(l1l2l3, l4)
        list.append(l1l2l3l4)
    # add the term in list
    operator = list[0]
    for i in range(len(list)-1):
        operator = operator_add(operator, list[i+1])
    for i in range(0, len(operator), 2):
        operator[i] = U*operator[i]
    return operator



Majorana = get_majorana('FH_10_ori.txt')
creation_up, creation_down, annihilation_up, annihilation_down = majorana_to_qubit_operator(Majorana)
first_operator = first_term(creation_up, creation_down, annihilation_up, annihilation_down)
second_operator = second_term(creation_up, creation_down, annihilation_up, annihilation_down)
Hamiltonian = operator_add(first_operator, second_operator)
print(Hamiltonian)


# Create a dictionary to map the strings to their corresponding complex numbers
dict = {}
for i in range(0, len(Hamiltonian), 2):
    complex_number = Hamiltonian[i]
    string = Hamiltonian[i + 1]
    dict[string] = complex_number
# Create a list of formatted strings
formatted_Hamiltonian = []
for string, complex_number in dict.items():
    formatted_Hamiltonian.append(f'{complex_number.real:+.15f} * {string}')
# Sort the formatted data alphabetically by string
formatted_Hamiltonian.sort()
# Write the formatted data to a text file
with open('FH_10.txt', 'w') as file:
    for line in formatted_Hamiltonian:
        file.write(line + '\n')

print("File 'output.txt' has been created with the desired format.")
