import os

path_name = '/Users/huhu/Documents/肌肉干细胞/代码/U-Net代码/U-Net-master/Unet/dataset/label/'
i = 0
for item in os.listdir(path_name):
    os.rename(os.path.join(path_name, item), os.path.join(path_name, str(i) + '_label.png'))
    i += 1

