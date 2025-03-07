import matplotlib.pyplot as plt

# using this function to process the row data of CNOT_numses and acc_reses
def data_process(CNOT_numses, acc_reses, step, alpha):
    num = len(CNOT_numses)
    group_CNOT_numses = []
    group_acc_reses = []

    for i in range(num):
        CNOT = CNOT_numses[i]
        acc = acc_reses[i]

        maximum = max(CNOT)
        minimum = min(CNOT)
        k_max = int((maximum + step - 0.00001) // step)
        k_min = int(minimum // step)
        rounded_max = k_max * step
        rounded_min = k_min * step

        group_CNOT = [k*step+step/2 for k in range(k_min, k_max)]
        group_acc_0 = [[] for k in range(k_min, k_max)]

        num_ = len(CNOT)
        for j in range(num_):
            k = CNOT[j]
            k = int(k // step -k_min)
            group_acc_0[k].append(acc[j])


        group_acc = [sum(group_acc_0[k-k_min])/len(group_acc_0[k-k_min]) if len(group_acc_0[k-k_min]) != 0 else 0 for k in range(k_min, k_max)]
 
        count = 0
        for j in range(len(group_acc)):
            if group_acc[j-count] == 0:
                group_acc.pop(j-count)
                group_CNOT.pop(j-count)
                count = count+1

        num_remove = int(len(group_acc) * alpha)
        if num_remove > 0:
            group_acc = group_acc[num_remove:-num_remove]
            group_CNOT = group_CNOT[num_remove:-num_remove]


        group_CNOT_numses.append(group_CNOT)
        group_acc_reses.append(group_acc)

    return group_CNOT_numses, group_acc_reses

CNOT_numses = [
[4004, 6104, 8176, 10278, 12482, 14398, 16526, 4150, 6292, 8154, 10200, 12396, 14420, 16264],
[3146, 4762, 6306, 7840, 9554, 11302, 12556, 3124, 4806, 6324, 7864, 9644, 10962, 12530],
[3186, 4946, 6322, 7830, 9480, 11050, 12826, 3092, 4910, 6164, 8106, 9696, 11160, 12550]
]

acc_reses = [
[0.9723138647850239, 0.9783855231634778, 0.9926236271893086, 0.9919114710367284, 0.9946557882252938, 0.9894517064112534, 0.995446265372187, 0.9868422164272133, 0.9837751144388327, 0.9880579151278988, 0.9830416635132102, 0.9874337433099857, 0.9958912134211293, 0.9921576977567629],
[0.9703627203582591, 0.9880319244287186, 0.9841008388081993, 0.9831926833015738, 0.9914720660649718, 0.9909443863085231, 0.9932401207356373, 0.9810637840011683, 0.9899426718670468, 0.9893576504454578, 0.9903655064271306, 0.9891207476268817, 0.9940471704844636, 0.9939185980141572],
[0.9781292846752291, 0.9841656704777609, 0.9935270557368687, 0.9872576642457533, 0.9954710006820767, 0.9918782185468208, 0.9949669243658066, 0.9662606268901229, 0.9858046222203265, 0.9875968148224735, 0.9925165791509268, 0.9931933880595102, 0.9941927830111321, 0.989686246452079]
]


if len(CNOT_numses) == 3:
    plt.scatter(acc_reses[0], CNOT_numses[0], color='b', label='1.0, 0.0, 0.0', marker='x')
    plt.scatter(acc_reses[1], CNOT_numses[1], color='g', label='0.4, 0.6, 0.0', marker='^')
    plt.scatter(acc_reses[2], CNOT_numses[2], color='y', label='0.4, 0.3, 0.3', marker='o')
    plt.legend(loc='upper left', fontsize=16)

plt.show()
# plt.savefig('Pauli_HF_f' + '\\photo' + '1s+' + '.png')