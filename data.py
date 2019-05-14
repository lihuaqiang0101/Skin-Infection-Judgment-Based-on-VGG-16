import os
from PIL import Image

if not os.path.exists('traindata'):#如果训练目录不存在就创建一个
    os.mkdir('traindata')
if not os.path.exists('testdata'):#如果测试目录不存在就创建一个
    os.mkdir('testdata')

train_num_0 = 0#训练数据负样本的个数
test_num_0 = 0#测试数据负样本的个数
train_num_1 = 0#训练数据正样本的个数
test_num_1 = 0#测试数据正样本的个数
datasets = os.listdir('train')
#遍历训练集中的每一张图片
for index in range(len(datasets)):
    images = os.listdir(r'train\{}'.format(datasets[index]))
    for image in images:
        try:
            img = Image.open(r'train\{}\{}'.format(datasets[index],image))
            img = img.resize((224,224))#将原始图片缩放成224x224大小
            if train_num_0 <= int(len(images)*0.7):#将原始数据中70%用作训练集
                if datasets[index] == 'Parasitized':#如果是负样本就标记为0
                    img.save(r'traindata\{}.0.jpg'.format(train_num_0))
                    train_num_0 += 1
            else:#原始数据中30%用作测试集
                if datasets[index] == 'Parasitized':#如果是负样本就标记为0
                    img.save(r'testdata\{}.0.jpg'.format(test_num_0))
                    test_num_0 += 1
            if train_num_1 <= int(len(images)*0.7):#正样本标记为1
                if datasets[index] == 'Uninfected':
                    img.save(r'traindata\{}.1.jpg'.format(train_num_0+train_num_1))
                    train_num_1 += 1
            else:#正样本标记为1
                if datasets[index] == 'Uninfected':
                    img.save(r'testdata\{}.1.jpg'.format(test_num_0+test_num_1))
                    test_num_1 += 1
        except:
            pass