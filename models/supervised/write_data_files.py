import sys
import h5py
import numpy as np
import pdb

sys.path.insert(0, '../')
from ecg_AAAI.parse_dataset.readECG import loadECG, generate_all_test_files
from scipy.io import loadmat
patient_outcomes = loadmat("./datasets/patient_outcomes.mat")['outcomes'] 
id_death_map = {x[0]: x[7] for x in patient_outcomes}

train_normal_ids = [1.0, 10.0, 100.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 101.0, 1010.0, 1011.0, 1012.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 102.0, 1020.0, 1021.0, 1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 103.0, 104.0, 1047.0, 1048.0, 1049.0, 105.0, 1051.0, 1052.0, 1053.0, 1054.0, 1055.0, 1056.0, 1057.0, 1058.0, 1059.0, 106.0, 1060.0, 1061.0, 1062.0, 1063.0, 1064.0, 1065.0, 1066.0, 1067.0, 1069.0, 107.0, 1070.0, 1071.0, 1072.0, 1073.0, 1075.0, 1076.0, 108.0, 1086.0, 1088.0, 1089.0, 109.0, 1090.0, 1091.0, 1092.0, 1093.0, 1094.0, 1095.0, 11.0, 110.0, 1104.0, 1105.0]
train_death_ids = [10171.0, 10239.0, 1031.0, 10395.0, 10422.0, 1050.0, 10502.0, 1087.0, 1125.0, 1136.0, 1175.0, 1189.0, 1197.0, 1241.0, 1273.0, 1413.0, 1515.0, 1630.0, 1646.0, 1698.0, 1710.0, 1720.0, 1726.0, 1741.0, 1763.0, 1816.0, 1921.0, 1931.0, 1938.0, 2036.0, 2236.0, 2252.0, 2317.0, 2343.0, 2388.0, 2438.0, 2484.0, 2617.0, 2628.0, 2635.0, 2718.0, 2785.0, 2899.0, 3046.0, 3053.0, 3216.0, 3232.0, 3245.0, 3277.0, 3319.0, 3348.0, 3350.0, 3525.0, 3541.0, 3548.0, 3570.0, 3596.0, 3617.0, 3630.0, 3792.0, 3797.0, 3865.0, 3875.0, 3991.0, 4000.0, 4128.0, 4153.0, 4210.0, 4249.0, 4353.0, 4381.0, 4383.0, 4467.0, 4516.0, 4659.0, 4740.0, 4763.0, 48.0, 4844.0, 4879.0]
test_normal_ids =  [1106.0, 1107.0, 1109.0, 111.0, 1110.0, 1111.0, 1112.0, 1113.0, 1114.0, 1115.0, 1116.0, 1117.0, 1118.0, 1119.0, 112.0, 1120.0, 1121.0, 1123.0, 1124.0, 1126.0, 1127.0, 1128.0, 1129.0, 113.0, 1130.0, 1131.0, 1133.0, 1134.0, 1135.0, 1138.0, 1139.0, 114.0, 1140.0, 1141.0, 1142.0, 1143.0, 1146.0, 1147.0, 1148.0, 1149.0, 115.0, 116.0, 117.0, 1171.0, 1176.0, 1179.0, 118.0, 1182.0, 1188.0, 119.0, 12.0, 120.0, 121.0, 123.0, 1232.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 13.0, 130.0, 131.0, 132.0, 133.0, 1332.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 1417.0, 144.0, 1463.0, 15.0, 1547.0, 1554.0, 16.0, 1616.0, 1644.0, 1687.0, 17.0, 1723.0, 1725.0, 1727.0, 1731.0, 1733.0, 1746.0, 18.0, 1804.0, 19.0, 1902.0, 1936.0, 1992.0, 20.0, 2004.0, 21.0, 2152.0, 22.0, 2234.0, 2267.0, 2290.0, 23.0, 2390.0, 24.0, 25.0, 2507.0, 2555.0, 2556.0, 2592.0, 26.0, 27.0, 2705.0, 2779.0, 28.0, 2812.0, 2831.0, 2832.0, 2833.0, 2841.0, 2844.0, 2865.0, 29.0, 2965.0, 2966.0, 2987.0, 3.0, 30.0, 3010.0, 3014.0, 3016.0, 3018.0, 3040.0, 3054.0, 3063.0, 3077.0, 31.0, 3181.0, 32.0, 3212.0, 3217.0, 3224.0, 3239.0, 3249.0, 3255.0, 3266.0, 3268.0, 33.0, 34.0, 3445.0, 3495.0, 3522.0, 3561.0, 3566.0, 3575.0, 36.0, 3621.0, 3671.0, 3679.0, 368.0, 3680.0, 369.0, 37.0, 370.0, 3704.0, 371.0, 3717.0, 372.0, 3722.0, 3723.0, 3727.0, 373.0, 374.0, 3748.0, 375.0, 3751.0, 376.0, 377.0, 3779.0, 3787.0, 379.0, 38.0, 3801.0, 381.0, 382.0, 3861.0, 39.0, 3938.0, 3950.0, 3957.0, 4.0, 40.0, 4002.0, 4010.0, 4018.0, 4073.0, 4096.0, 4132.0, 4169.0, 4171.0, 4174.0, 4193.0, 42.0, 4236.0, 4240.0, 4245.0, 4248.0, 4257.0, 4263.0, 43.0, 437.0, 439.0, 4391.0, 44.0, 4407.0, 441.0, 4419.0, 442.0, 4442.0, 4450.0, 4468.0, 45.0, 4517.0, 4523.0, 4585.0, 46.0, 47.0, 4741.0, 4747.0, 4755.0, 4756.0, 4766.0, 4840.0, 4841.0, 4869.0, 4882.0, 49.0, 4917.0, 494.0, 495.0, 4954.0, 497.0, 4975.0, 4976.0, 498.0, 4980.0, 4982.0, 499.0, 4992.0, 5.0, 50.0, 500.0, 5004.0, 502.0, 503.0, 5065.0, 5072.0, 5083.0, 5088.0, 5098.0, 51.0, 5106.0, 5120.0, 5123.0, 5144.0, 5157.0, 5171.0, 52.0, 5235.0, 528.0, 529.0, 53.0, 54.0, 5429.0, 5444.0, 5480.0, 5482.0, 5488.0, 55.0, 5503.0, 5513.0, 5523.0, 5531.0, 5533.0, 5545.0, 5563.0, 5582.0]
test_normal_supplement = [5593.0, 5601.0, 5605.0, 5610.0, 5629.0, 5636.0, 564.0, 5644.0, 5646.0, 5649.0, 5655.0, 567.0, 5681.0, 5688.0, 57.0, 5703.0, 5713.0, 5717.0, 5719.0, 5732.0, 5733.0, 5745.0, 5747.0, 5755.0, 58.0, 584.0, 5882.0, 59.0, 5903.0, 5924.0, 6.0, 60.0, 6002.0, 6086.0, 61.0, 6102.0, 6114.0, 6120.0, 613.0, 614.0, 615.0, 618.0, 62.0, 6240.0, 629.0, 63.0, 64.0, 647.0, 65.0, 651.0, 652.0, 655.0, 656.0, 657.0, 658.0, 659.0, 66.0, 660.0, 661.0, 662.0, 663.0, 664.0, 668.0, 67.0, 68.0, 680.0, 681.0, 687.0, 688.0, 69.0, 690.0, 691.0, 694.0, 695.0, 696.0, 697.0, 698.0, 699.0, 7.0, 70.0, 700.0, 703.0, 704.0, 705.0, 706.0, 71.0, 719.0, 72.0, 720.0, 721.0, 722.0, 723.0, 724.0, 725.0, 73.0, 735.0, 737.0, 738.0, 739.0, 74.0, 740.0, 741.0, 742.0, 744.0, 745.0, 746.0, 747.0, 748.0, 749.0, 75.0, 750.0, 751.0, 752.0, 753.0, 754.0, 755.0, 756.0, 757.0, 759.0, 761.0, 762.0, 763.0, 764.0, 765.0, 769.0, 770.0, 771.0, 772.0, 773.0, 774.0, 775.0, 776.0, 777.0, 778.0, 779.0, 78.0, 780.0, 781.0, 782.0, 783.0, 784.0, 785.0, 786.0, 787.0, 788.0, 789.0, 79.0, 790.0, 798.0, 799.0, 8.0, 80.0, 800.0, 801.0, 802.0, 803.0, 804.0, 806.0, 807.0, 808.0, 81.0, 810.0, 811.0, 812.0, 813.0, 814.0, 815.0, 817.0, 818.0, 819.0, 82.0, 820.0, 821.0, 822.0, 824.0, 825.0, 826.0, 827.0, 829.0, 830.0, 831.0, 833.0, 834.0, 835.0, 836.0, 837.0, 838.0, 839.0, 84.0, 840.0, 841.0, 842.0, 848.0, 852.0, 853.0, 855.0, 856.0, 857.0, 858.0, 859.0, 860.0, 861.0, 862.0, 863.0, 864.0, 865.0, 866.0, 867.0, 868.0, 869.0, 87.0, 870.0, 871.0, 873.0, 874.0, 875.0, 876.0, 878.0, 88.0, 880.0, 881.0, 882.0, 883.0, 884.0, 886.0, 887.0, 888.0, 89.0, 890.0, 891.0, 892.0, 893.0, 894.0, 895.0, 897.0, 9.0, 90.0, 900.0, 901.0, 902.0, 903.0, 904.0, 906.0, 907.0, 91.0, 914.0, 915.0, 916.0, 917.0, 92.0, 920.0, 921.0, 922.0, 923.0, 924.0, 925.0, 926.0, 927.0, 928.0, 929.0, 93.0, 930.0, 931.0, 932.0, 933.0, 934.0, 935.0, 936.0, 937.0, 939.0, 940.0, 941.0, 942.0, 943.0, 944.0, 945.0, 946.0, 947.0, 95.0, 950.0, 951.0, 953.0, 954.0, 955.0, 956.0, 957.0, 958.0, 959.0, 96.0, 960.0, 962.0, 963.0, 964.0, 965.0, 966.0, 967.0, 968.0, 969.0, 97.0, 970.0, 971.0, 972.0, 973.0, 974.0, 975.0, 976.0, 978.0, 979.0, 98.0, 980.0, 981.0, 982.0, 983.0, 984.0, 985.0, 986.0, 988.0, 989.0, 99.0, 990.0, 992.0, 993.0, 994.0, 995.0, 996.0, 998.0, 999.0]
test_normal_ids = test_normal_ids + test_normal_supplement

test_death_ids = [5133.0, 5180.0, 5187.0, 5194.0, 5506.0, 5616.0, 5687.0, 5908.0, 5964.0, 6160.0, 6311.0, 760.0, 832.0, 854.0, 96.0]
all_test_ids = np.loadtxt('/home/divyas/ecg_AAAI/datasets/ecg_data/hard_drive_test_ids')
pdb.set_trace()
#loadECG(train_normal_ids, train_death_ids, test_normal_ids, test_death_ids)
generate_all_test_files(all_test_ids, id_death_map)
