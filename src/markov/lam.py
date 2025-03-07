### this function calculate the lambda of each molecule

def get_pauli(path: str):
    file = open(file=path, mode="r", encoding='utf-8')
    file = file.readlines()
    res = []
    for line in file:
        if len(line) < 2:
            continue
        line = line.split(' ')
        print(line)
        val = int(float(line[1]) * 10000 + 0.5)
        if val == 0:
            continue
        val = float(val) / 10000
        print(line)
        # print(line[3][:-1])
        if line[0] == '+':
            res.append([line[3][:-1], val])
        if line[0] == '-':
            res.append([line[3][:-1], -val])
    print(res)
    print(len(res))
    return res



def get_CNOT_matrix(input_pauli):
    string_num = len(input_pauli)
    # string length stand for the total qubit in the problem
    string_len = len(input_pauli[0][0])
    res = [[0 for i in range(string_num)] for i in range(string_num)]
    for ii in range(string_num):
        for jj in range(string_num):
            count = 0
            string_i = input_pauli[ii][0]
            string_j = input_pauli[jj][0]
            for i in range(string_len):
                if string_i[i] != string_j[i]:
                    count = count + 1
                    if string_i[i] != 'I' and string_j[i] != 'I':
                        count = count + 1
            res[ii][jj] = count
    return res


def get_single_q_matrix(input_pauli):
    string_num = len(input_pauli)
    string_len = len(input_pauli[0][0])
    res = [[0 for i in range(string_num)] for i in range(string_num)]
    XY_count = [0 for i in range(string_num)]
    for ii in range(string_num):
        string_i = input_pauli[ii][0]
        for i in range(string_len):
            if string_i[i] == 'X':
                XY_count[ii] += 1
            if string_i[i] == 'Y':
                XY_count[ii] += 1
    for ii in range(string_num):
        for jj in range(string_num):
            count = 0
            string_i = input_pauli[ii][0]
            string_j = input_pauli[jj][0]
            for i in range(string_len):
                if string_i[i] == string_j[i]:
                    if string_i[i] == 'X' or string_i[i] == 'Y':
                        count = count + 1
            res[ii][jj] = XY_count[ii] + XY_count[jj] - count * 2
    return res



input_pauli = get_pauli('_Pauli_string_BeH2_unf')
# A = get_CNOT_matrix(input_pauli)
# B = get_single_q_matrix(input_pauli)
# print(A)
# print(B)
# # print(A)
# # print("s")
# # print(B)
# exceed_count = 0

# Iterate through the matrices and compare elements
# for i in range(len(A)):
#     for j in range(len(A[0])):
#         if B[i][j] > A[i][j]:
#             exceed_count += 1

# Print the result
# print("Number of elements in matrix B that exceed their counterparts in matrix A:", exceed_count)
string_num = len(input_pauli)
lamda = 0.0
for string in input_pauli:
    lamda += max(string[1], -string[1])

print(lamda)