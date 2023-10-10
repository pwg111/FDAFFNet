import csv
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
def get_iou_list(filename):
    with open(filename) as f:  # 打开这个文件，并将结果文件对象存储在f中
        reader = csv.reader(f)  # 创建一个阅读器reader
        header_row = next(reader)  # 返回文件中的下一行
        epoch, train_iou, val_iou = [], [0.0], [0.0]  # 声明存储日期，最值的列表
        i = 1
        for row in reader:
            epoch.append(i)  # 存储日期
            tiou = float(format(float(row[1]), '.4f')) # 将字符串转换为数字
            train_iou.append(tiou)  # 存储温度最大值
            viou = float(format(float(row[2]), '.4f'))
            val_iou.append(viou)  # 存储温度最小值
            i = i + 1
        return train_iou, val_iou
def get_mine_iou_list(filename):
    with open(filename) as f:  # 打开这个文件，并将结果文件对象存储在f中
        reader = csv.reader(f)  # 创建一个阅读器reader
        header_row = next(reader)  # 返回文件中的下一行
        epoch, train_iou, val_iou = [], [0.0], [0.0]  # 声明存储日期，最值的列表
        i = 1
        for row in reader:
            epoch.append(i)  # 存储日期
            tiou = float(format(float(row[1])/100, '.4f')) # 将字符串转换为数字
            train_iou.append(tiou)  # 存储温度最大值
            viou = float(format(float(row[2])/100, '.4f'))
            val_iou.append(viou)  # 存储温度最小值
            i = i + 1
        return train_iou, val_iou
# 读取CSV文件数据

epoch = [_ for _ in range(61)]
diff_filename = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\diff\loss.csv'
diff_train_iou,diff_val_iou = get_iou_list(diff_filename)
DSIFN_filename = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\DSIFN\loss.csv'
DSIFN_train_iou,DSIFN_val_iou = get_iou_list(DSIFN_filename)
Mine_filename = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\Mine_l_C2\loss.csv'
Mine_train_iou,Mine_val_iou = get_mine_iou_list(Mine_filename)
DSAM_filename = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\DSAM\loss.csv.csv'
DSAM_train_iou,DSAM_val_iou = get_iou_list(DSAM_filename)
# 根据数据绘制图形

#创建数据

#创建figure窗口，figsize设置窗口的大小
plt.figure(num=3, figsize=(8, 5))
#画曲线1
plt.plot(epoch, diff_val_iou[0:61], label='Siam_diff')
#画曲线2
plt.plot(epoch, DSIFN_val_iou[0:61], label='DSIFN')
plt.plot(epoch[0:31], DSAM_val_iou[0:31], label='DSAM')
plt.plot(epoch, Mine_val_iou[0:61], label='Mine')
plt.legend()
#设置坐标轴范围
plt.xlim((0, 60))
plt.ylim((0, 1))
#设置坐标轴名称
plt.xlabel('epoch')
plt.ylabel('val_iou')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 60, 10)
my_y_ticks = np.arange(0, 1, 0.1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

#显示出所有设置
plt.show()
