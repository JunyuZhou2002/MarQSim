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
[3272, 4944, 6596, 8018, 9902, 11264, 13168, 3120, 4916, 6534, 8330, 9800, 11430, 13216, 3274, 4974, 6488, 8144, 9966, 11414, 13310, 3248, 4816, 6696, 8324, 9948, 11572, 13036],
[2596, 3824, 4960, 6358, 7844, 9286, 10046, 2424, 3806, 4892, 6544, 7540, 8992, 10194, 2544, 3870, 5096, 6458, 7680, 8854, 9894, 2566, 3876, 5164, 6318, 7650, 8874, 10082],
[2530, 3824, 5098, 6202, 7632, 8980, 10116, 2520, 3804, 5146, 6354, 7622, 8858, 9920, 2464, 3742, 5302, 6284, 7904, 8676, 10226, 2528, 3700, 5154, 6378, 7556, 8910, 10060]
]

acc_reses = [
[0.9813717037031234, 0.9904909986365413, 0.9903800496196089, 0.9929147982061879, 0.9922937755560205, 0.9885045891539179, 0.9965136051216343, 0.9641940411392098, 0.9846397042280345, 0.9942720533459029, 0.9911512161098021, 0.9906553031075201, 0.9926865136694594, 0.9919069410779671, 0.9753065234488231, 0.9811397225834767, 0.9907690286946347, 0.9868699917465055, 0.9910566857742898, 0.9952470661003661, 0.9956822408848519, 0.9875538946510165, 0.9904271963219059, 0.9813517790402162, 0.9941208364127235, 0.9930285280841747, 0.9947392940560914, 0.9913711550131724],
[0.9890283355210125, 0.9818897334982026, 0.9840050960681419, 0.9941163069666535, 0.9919458098970284, 0.9944226754006743, 0.9880922445295862, 0.9884100576784656, 0.9853693903894546, 0.986204803243307, 0.9933686853728815, 0.993649840366009, 0.992827061383577, 0.9953117399653878, 0.968056869910027, 0.98816073115214, 0.9934496324364429, 0.9905778198239091, 0.9935287870042093, 0.9936085941417826, 0.9824917707444929, 0.9745413115587652, 0.980520568840145, 0.9891188801625496, 0.9924288445196358, 0.991353585637091, 0.9973382955110043, 0.9948523656519983],
[0.9731581085792409, 0.9921974049138778, 0.9897710288612499, 0.9933460973360537, 0.988991803617452, 0.9965708853704733, 0.9962739944629748, 0.9827058770589122, 0.9869626771276276, 0.9921980568467016, 0.9943348169330717, 0.991620779704134, 0.992837607906256, 0.9899817184439039, 0.9561172610363017, 0.9895685596230309, 0.9899315843118942, 0.9828976421482397, 0.9952347881016215, 0.992095477933452, 0.9937082546715115, 0.9556373428869337, 0.986314252723239, 0.9920121045086543, 0.9885907769249715, 0.9925109163546396, 0.992195073624366, 0.9869368686192882]
]


if len(CNOT_numses) == 3:
    plt.scatter(acc_reses[0], CNOT_numses[0], color='b', label='1.0, 0.0, 0.0', marker='x')
    plt.scatter(acc_reses[1], CNOT_numses[1], color='g', label='0.4, 0.6, 0.0', marker='^')
    plt.scatter(acc_reses[2], CNOT_numses[2], color='y', label='0.4, 0.3, 0.3', marker='o')
    plt.legend(loc='upper left', fontsize=16)

plt.show()
# plt.savefig('Pauli_HF_f' + '\\photo' + '1s+' + '.png')