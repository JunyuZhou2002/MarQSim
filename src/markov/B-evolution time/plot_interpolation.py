import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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


### Na+ pis
CNOT_numses = [
[52854, 79206, 105546, 130892, 159148, 183190, 210592, 52330, 78646, 105406, 131966, 159386, 183830, 210366, 52426, 78164, 104672, 131134, 159432, 183312, 211306, 52910, 78656, 104972, 130920, 159348, 184010, 210138, 52878, 78370, 104760, 131622, 158672, 183432, 210024, 52976, 78570, 105140, 131372, 159648, 184186, 209916, 52872, 78462, 104830, 131130, 159440, 182896, 211420, 52804, 78880, 105452, 131860, 160154, 184062, 210406, 52290, 78436, 105512, 131678, 158920, 184582, 210594, 52740, 78776, 104940, 131374, 158806, 183280, 211278, 52500, 78986, 105160, 131170, 159790, 184918, 211466, 52498, 78780, 104944, 131500, 159052, 184182, 210960, 51960, 78226, 105692, 132076, 159628, 184464, 210816, 52540, 78410, 105200, 132376, 159604, 185260, 211944, 53050, 78660, 104686, 131272, 159900, 183752, 209336, 52680, 78248, 106162, 131978, 159350, 183808, 210146, 52810, 79040, 104948, 131780, 159594, 183942, 210124, 52810, 79276, 105264, 131974, 159534, 184292, 211418, 53208, 78992, 104870, 131480, 159392, 184092, 210238, 52214, 78028, 105286, 131706, 159160, 185042, 210396],
[40416, 60096, 81300, 101696, 123914, 142590, 161946, 40650, 60960, 81038, 101644, 123580, 142044, 161904, 40672, 60240, 80986, 101180, 122812, 141912, 162904, 40348, 60540, 81446, 101860, 122732, 142354, 162200, 40698, 60784, 81628, 102126, 122758, 142084, 163170, 41042, 61026, 81070, 101742, 123018, 142632, 162164, 40614, 60340, 80502, 101276, 123074, 141982, 162162, 40252, 60602, 81294, 101842, 123194, 142436, 162232, 40550, 60834, 80890, 101532, 123040, 142284, 162092, 40790, 60292, 81040, 101944, 122984, 142170, 162186, 40652, 60232, 81062, 101720, 122534, 141778, 162106, 40300, 60810, 80984, 100988, 123722, 142198, 162162, 40650, 61024, 81182, 101682, 123758, 141238, 161944, 40942, 60352, 81338, 101864, 123062, 142014, 162746, 40152, 60948, 81528, 101624, 123204, 142542, 162868, 40554, 60484, 81262, 101118, 123856, 143096, 161786, 40302, 60628, 80830, 102298, 122866, 142404, 162614, 40598, 60520, 80628, 101794, 123880, 142068, 161532, 40894, 60972, 81302, 101928, 122832, 141670, 162236, 40672, 60208, 81366, 101760, 123360, 142712, 162890],
[40510, 60198, 81794, 102352, 124104, 142742, 162540, 40782, 60884, 81374, 101412, 123798, 142260, 162738, 40774, 60646, 80680, 101786, 122386, 142206, 162186, 40548, 60882, 81588, 101022, 123494, 142916, 162946, 40720, 60544, 80774, 101796, 123176, 142832, 162450, 40620, 60896, 81460, 101310, 122998, 142422, 162490, 40656, 60372, 80734, 101626, 122980, 142246, 163022, 40724, 61384, 81282, 101626, 123130, 142572, 162934, 41136, 60906, 81476, 101650, 123398, 142444, 162872, 41270, 61134, 81630, 102182, 124062, 142862, 162650, 40796, 61116, 81930, 101460, 123750, 142358, 162832, 40730, 60250, 81282, 101436, 123572, 142736, 162752, 40734, 60724, 81420, 101570, 123100, 142826, 162240, 40546, 61230, 81124, 101574, 122620, 142902, 163190, 40990, 60974, 82170, 102166, 122696, 142368, 162910, 41286, 60682, 81832, 101318, 124166, 143406, 163314, 40998, 61156, 81918, 101192, 122784, 142036, 162564, 40790, 61248, 81356, 100938, 122824, 142750, 163026, 40956, 61060, 81588, 102626, 124194, 142382, 162844, 40610, 60412, 81422, 101810, 124200, 141940, 163066]
]

acc_reses = [
[0.9866213803689756, 0.9825083322468622, 0.9913410575979187, 0.9901609492341842, 0.9967767596693385, 0.993532955656851, 0.995439946730341, 0.9880348544011746, 0.9852560825335666, 0.9788492446109011, 0.9923080206655238, 0.9963272811924203, 0.9938027877575571, 0.9958728552809959, 0.984553277495777, 0.991516298699366, 0.9845438003891056, 0.9831607360865062, 0.9933908036652818, 0.9911370896540982, 0.993302583714355, 0.987655302178881, 0.9859073184763671, 0.9832948215042512, 0.995394825976634, 0.9930675633533564, 0.9926450253578981, 0.9975073934202937, 0.9901104651005999, 0.9878350641612351, 0.9859476685865229, 0.9916877201079594, 0.9934628027689575, 0.9906438602016014, 0.9946555512442251, 0.9724337781487573, 0.9677785316696035, 0.9893319614024432, 0.9849482192175278, 0.991807580407177, 0.9951874017837345, 0.9950134158182723, 0.9793476926174446, 0.9814810196222359, 0.9879761338320582, 0.991062416716022, 0.996033489915937, 0.9942692521471961, 0.9939612179648717, 0.965872050055692, 0.9733776528491692, 0.9825675487569285, 0.993687808691489, 0.9955410970074522, 0.9927957670750417, 0.9940093773152829, 0.9745206688030716, 0.9805987477328978, 0.989794492967282, 0.9952805069389423, 0.9926822195214865, 0.991956532614076, 0.9923230410804228, 0.9840705463802412, 0.9896331460909967, 0.991618276487565, 0.980572532810398, 0.9887676846557042, 0.9912729357280492, 0.9913967256093384, 0.9767356233970564, 0.9761270880991245, 0.9825065804528573, 0.9925813695877406, 0.9928371312379566, 0.9823226335775102, 0.9953918464137075, 0.9693781161532427, 0.9873548925401361, 0.9877628100797489, 0.9935143608049, 0.9920284243322398, 0.9901856802208436, 0.9908077260777447, 0.9571165384679573, 0.9867350506190955, 0.9810951948781802, 0.9889958508337837, 0.9924792070333858, 0.9916351439607941, 0.9951393605562018, 0.9699625547255859, 0.982267106630915, 0.9890345936031573, 0.9888145936851943, 0.9941222081531306, 0.9881002316166351, 0.9946609751658575, 0.9675111793484092, 0.9774544704287875, 0.9870141385131568, 0.992102924559745, 0.9916576430286952, 0.9881675249064901, 0.9934986829763021, 0.9809488675843239, 0.9832367900485667, 0.9874120070224455, 0.9920045738233536, 0.9959232809412302, 0.9924151910233225, 0.9930280488528623, 0.978309484399738, 0.993275582954033, 0.9871771941606755, 0.992798447621993, 0.995162904599952, 0.9923858645731239, 0.9942666888285282, 0.9633329186874335, 0.9845392274664448, 0.992602829697936, 0.9910997208288194, 0.995482921720763, 0.993273735497491, 0.9929031136452398, 0.9815401442595035, 0.9873457551411855, 0.9918954978477428, 0.9840165165964339, 0.9945470096440815, 0.9895427841716726, 0.9948777556595358, 0.9644599130856084, 0.9879879952567819, 0.9895966189526971, 0.9864033642493368, 0.9932617546116912, 0.9914566289367283, 0.9947461494264586],
[0.9797204326337463, 0.988001526830606, 0.987642614833645, 0.9937852033435016, 0.9943886996094197, 0.9878860120600337, 0.9933055642452426, 0.9755651433615977, 0.9710852658865012, 0.9839705583029541, 0.983320202940074, 0.9917066710229253, 0.9917851690779312, 0.9929201014130115, 0.9623461078910596, 0.9837755645145232, 0.9912780564596108, 0.9831237044633098, 0.9953961968543114, 0.9869214812577337, 0.9953063137278249, 0.9785333342483968, 0.9821792857807042, 0.992005705872604, 0.9943394064910356, 0.9930660747111404, 0.9931331876176113, 0.9919756692632674, 0.956109135227237, 0.9870057706387495, 0.9913499357528317, 0.9913232342429892, 0.9888195239665882, 0.9825467476404739, 0.9922230310458526, 0.9596216832061693, 0.9855766296925014, 0.9861905846337008, 0.9880999433911334, 0.9880446791689099, 0.9928214557080157, 0.992927374491623, 0.9713769651603807, 0.9820488489914935, 0.9856517489783831, 0.9946792721861937, 0.9919298296973444, 0.9937068822567174, 0.9946145419347976, 0.9812977997408129, 0.9870046707578716, 0.9891955334556344, 0.9945464134304499, 0.9949101813697723, 0.9910803439462553, 0.9926949151532404, 0.9637943285352399, 0.98299441858136, 0.992183105493319, 0.9914849411625146, 0.9894395708404629, 0.9887995745226273, 0.9963430764370697, 0.9684731113309126, 0.9673634947112355, 0.9845558406483866, 0.985536947912035, 0.9903700953939323, 0.9921574024386526, 0.9974823595829223, 0.975575506874342, 0.9905016453979087, 0.9901381310157874, 0.9828636273023238, 0.9783424139657512, 0.9930009250586312, 0.9958727359891517, 0.9844638145488304, 0.9916849137935729, 0.970119291561286, 0.9932118667394054, 0.9861065745921876, 0.9927054049016405, 0.9943982958799711, 0.9784568558976725, 0.9938357473051033, 0.9852443367585412, 0.9913522617019562, 0.9891414046312725, 0.9877259736426293, 0.9893662998818552, 0.9688907980365906, 0.9833652399633105, 0.9800326926251379, 0.982487390836124, 0.9953622225907905, 0.9911276102357369, 0.9911690951968407, 0.9841273665150804, 0.990832269348296, 0.9895735459690519, 0.9915141467995631, 0.9951677920076973, 0.9878366005845661, 0.9850972089607138, 0.9790631446103546, 0.992420856129622, 0.9817279737080437, 0.9898391634710235, 0.9915854372264342, 0.9881911446827857, 0.9983780980694074, 0.9838570008818188, 0.9731895524040511, 0.9863364100656914, 0.9912313834680906, 0.9939286168362793, 0.9769258347349429, 0.9959272147637447, 0.9799553332106359, 0.9688317501201256, 0.9818525859445882, 0.9930408064491018, 0.9868769049690557, 0.9897657102437464, 0.9954128865513249, 0.9655479267746624, 0.9847673327392795, 0.9846486466846353, 0.9884610203320607, 0.9804201234348005, 0.9922451989618527, 0.9951962233182191, 0.9807332092083673, 0.9897561075159275, 0.9765891148529209, 0.9874993617568776, 0.9924666881853949, 0.9925666461114927, 0.9956803321986261],
[0.9802352666992042, 0.9884276991422031, 0.9934223302795344, 0.9887921976222875, 0.9945099951748019, 0.989999655498345, 0.9964320064308356, 0.9844249353857204, 0.9843008620656405, 0.9867088412773776, 0.9870597802951107, 0.9973649988695633, 0.9913699030896134, 0.9919930100752737, 0.9766182553735127, 0.9843821051618856, 0.9929595594087379, 0.9878272586660646, 0.9928022887599347, 0.9922450191752139, 0.9958212375718456, 0.975184801335326, 0.9921976925809279, 0.984351536547359, 0.9937506914143409, 0.9900721908185329, 0.9928445823654044, 0.9918226006346245, 0.9696535318839984, 0.9827000507721412, 0.9721931379926143, 0.9939523340549588, 0.9914593107134776, 0.9904385836764373, 0.9919703010834622, 0.9722528022721212, 0.989677796640308, 0.9932030460218186, 0.9897691019044405, 0.9958807855073658, 0.9925339222016404, 0.9935855869146988, 0.9752598406023144, 0.9879754951181654, 0.9822730749129759, 0.9929856676583888, 0.996007978987929, 0.9891551148846263, 0.9944207668432586, 0.9889601474236976, 0.9770613912033255, 0.9865150502686918, 0.9916877171660966, 0.9919693522613717, 0.9934297242778675, 0.9931824972087996, 0.9839466216054721, 0.9879191533728268, 0.9880142911967511, 0.9935914280651635, 0.9888174705815534, 0.9901659489090627, 0.9931905281180973, 0.9624480774467155, 0.988209558248045, 0.9906274082030065, 0.9951717257520647, 0.9928453639300395, 0.9891428947727194, 0.9942860010764656, 0.9852487817459815, 0.9900800586393308, 0.9842219082839224, 0.9879906807794347, 0.9956412912002103, 0.993357959648743, 0.9958206418109908, 0.9677115375837573, 0.978814913973889, 0.986953446829995, 0.9959823510224163, 0.993005873921371, 0.9887548691480571, 0.9942736647375237, 0.9800938973447723, 0.9701523402720227, 0.992528678124029, 0.9895296692883679, 0.9929584876688167, 0.9911534190549242, 0.9933765582626559, 0.9876390167319997, 0.9830446444789781, 0.9841012628948677, 0.985920548443457, 0.9900443083193232, 0.9922619462022085, 0.9960924387086432, 0.9849683661976234, 0.9838427420392403, 0.9816957774495755, 0.9929074650836855, 0.9930378283630534, 0.9902795557138377, 0.9949448109684308, 0.9586566194056255, 0.9895735979710596, 0.986287355462202, 0.9925127030639349, 0.9907666670330395, 0.9855695390203776, 0.9916052225873982, 0.9821035350285165, 0.9920749664975892, 0.9927604817424164, 0.9886721537110408, 0.9927168496618769, 0.993688887114324, 0.9933744072932972, 0.9764572986491958, 0.985234637859521, 0.9875018609423388, 0.9941017642145679, 0.9898158912897712, 0.9908659469367477, 0.9967908263704026, 0.9856130956734227, 0.980948717366325, 0.9889969256505515, 0.993302586302845, 0.9948560601635645, 0.9899927392351414, 0.9963676930196758, 0.9706909740032997, 0.981824122222139, 0.9916538783539683, 0.9754714370800918, 0.9956294314567808, 0.9919306090654971, 0.9923530821611798]
]

def drop_min_elements(nums):
    # Check if the list has at least 2 elements
    if len(nums) < 2:
        return nums  # Nothing to drop
    
    # Sort the list in ascending order
    sorted_nums = sorted(nums)
    
    # Remove the first two elements from the sorted list
    sorted_nums = sorted_nums[2:]
    
    # Create a new list with remaining elements
    result = [num for num in nums if num in sorted_nums]
    
    return result

# Define the model function a + b * e^(c * x)
def model_function(x, a, b, c):
    return a + np.exp(b*x+c)


acc_reses_new = []
CNOT_numses_new = []
for i, acc_rese in enumerate(acc_reses):
    acc_rese_new = []
    CNOT_nums_new = []
    for j, acc in enumerate(acc_rese):
        # if j % 18 > 12 or j % 18 == 3 or j %18 == 8:
        acc_rese_new.append(acc)
        CNOT_nums_new.append(CNOT_numses[i][j])
    acc_reses_new.append(acc_rese_new)
    CNOT_numses_new.append(CNOT_nums_new)


# clustering
acc_clusters_1 = [drop_min_elements(acc_reses_new[0][i::7]) for i in range(7)]
acc_final_1 = np.array([sum(acc_clusters_1[i])/len(acc_clusters_1[i]) for i in range(len(acc_clusters_1))])
acc_std_1 = [np.std(acc_clusters_1[i]) for i in range(len(acc_clusters_1))]
CNOT_clusters_1 = np.array([sum(CNOT_numses_new[0][i::7])/len(CNOT_numses_new[0][i::7]) for i in range(7)])
acc_clusters_2 = [drop_min_elements(acc_reses_new[1][i::7]) for i in range(7)]
acc_final_2 = np.array([sum(acc_clusters_2[i])/len(acc_clusters_2[i]) for i in range(len(acc_clusters_2))])
acc_std_2 = [np.std(acc_clusters_2[i]) for i in range(len(acc_clusters_2))]
CNOT_clusters_2 = np.array([sum(CNOT_numses_new[1][i::7])/len(CNOT_numses_new[1][i::7]) for i in range(7)])
acc_clusters_3 = [drop_min_elements(acc_reses_new[2][i::7]) for i in range(7)]
acc_final_3 = np.array([sum(acc_clusters_3[i])/len(acc_clusters_3[i]) for i in range(len(acc_clusters_3))])
acc_std_3 = [np.std(acc_clusters_3[i]) for i in range(len(acc_clusters_3))]
CNOT_clusters_3 = np.array([sum(CNOT_numses_new[2][i::7])/len(CNOT_numses_new[2][i::7]) for i in range(7)])


x_list = np.array([0.992, 0.9925, 0.993, 0.9935, 0.994])

params1, _ = curve_fit(model_function, acc_final_1, CNOT_clusters_1)
a_fit1, b_fit1, c_fit1 = params1

params2, _ = curve_fit(model_function, acc_final_2, CNOT_clusters_2)
a_fit2, b_fit2, c_fit2 = params2

params3, _ = curve_fit(model_function, acc_final_3, CNOT_clusters_3)
a_fit3, b_fit3, c_fit3 = params3



# y_fit1 = model_function(x_list, a_fit1, b_fit1, c_fit1)
# y_fit2 = model_function(x_list, a_fit2, b_fit2, c_fit2)
# y_fit3 = model_function(x_list, a_fit3, b_fit3, c_fit3)

y_fit1 = model_function(acc_final_1, a_fit1, b_fit1, c_fit1)
y_fit2 = model_function(acc_final_2, a_fit2, b_fit2, c_fit2)
y_fit3 = model_function(acc_final_3, a_fit3, b_fit3, c_fit3)


# calculate the reduction
reduce12 = (sum(y_fit1)-sum(y_fit2))/sum(y_fit1)
reduce13 = (sum(y_fit1)-sum(y_fit3))/sum(y_fit1)
print("gate reduction:", reduce12, reduce13)

# calculate the std reduction
reduce_std = (sum(acc_std_2)-sum(acc_std_3))/sum(acc_std_2)
print("std reduction:", reduce_std)

# original ave scatter
# plt.scatter(acc_final_1, CNOT_clusters_1, color='b', label='1.0, 0.0, 0.0', marker='x')
# plt.scatter(acc_final_2, CNOT_clusters_2, color='g', label='0.4, 0.6, 0.0', marker='^')
# plt.scatter(acc_final_3, CNOT_clusters_3, color='y', label='0.4, 0.3, 0.3', marker='o')

# with even x_axis
# plt.scatter(x_list, y_fit1, color='b', label='1.0, 0.0, 0.0', marker='x')
# plt.scatter(x_list, y_fit2, color='g', label='0.4, 0.6, 0.0', marker='^')
# plt.scatter(x_list, y_fit3, color='y', label='0.4, 0.3, 0.3', marker='o')

# using uneven x_axis
# plt.scatter(acc_final_1, y_fit1, color='b', label='1.0, 0.0, 0.0', marker='x')
# plt.scatter(acc_final_2, y_fit2, color='g', label='0.4, 0.6, 0.0', marker='^')
# plt.scatter(acc_final_3, y_fit3, color='y', label='0.4, 0.3, 0.3', marker='o')


list_x = [i+1 for i in range(len(acc_clusters_1))]
# standard deviation
# plt.scatter(list_x, acc_std_1, color='b', label='1.0, 0.0, 0.0', marker='x')
# plt.scatter(list_x, acc_std_2, color='g', label='0.4, 0.6, 0.0', marker='^')
# plt.scatter(list_x, acc_std_3, color='y', label='0.4, 0.3, 0.3', marker='o')

# original scatter
plt.scatter(acc_reses_new[0], CNOT_numses_new[0], color='b', label='1.0, 0.0, 0.0', marker='x')
plt.scatter(acc_reses_new[1], CNOT_numses_new[1], color='g', label='0.4, 0.6, 0.0', marker='^')
plt.scatter(acc_reses_new[2], CNOT_numses_new[2], color='y', label='0.4, 0.3, 0.3', marker='o')



plt.legend(loc='upper left', fontsize=16)
plt.show()
# plt.savefig('Pauli_OH-_f' + '\\pi6' + '\\photo' + '1_fit' + '.png')