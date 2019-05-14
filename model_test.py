import os
import torch
from torch import nn
from PIL import Image
import csv
from Net import VGG_16
import numpy as np
from torch.autograd import Variable

def Result(filepath):
    # 需要对测试结果输出，输出结果格式见result.csv
    imgs = os.listdir(filepath)#获取测试路径下的所有图片名
    f = open('result.csv','w',encoding='utf-8')#创建保存结果的文件
    filedname = ['filename','class']
    # 写入头部信息
    writer = csv.DictWriter(f,fieldnames=filedname)
    writer.writeheader()
    net = VGG_16()
    if torch.cuda.is_available():
        net = net.cuda()
    net.load_state_dict(torch.load('net_soft_params.pkl'))
    #用PIL打开路径下的每一个图片然后将大小统一转换为224x224
    IMGS = []
    imgname = []
    for img in imgs:
        Img = Image.open(os.path.join(filepath,img))
        Img = Img.resize((224,224))
        Img = np.array(Img)
        Img = np.transpose(Img,[2,0,1])#WHC->CHW
        IMGS.append(Img)
        imgname.append(img)
        if len(IMGS) == 2:
            IMGS = torch.Tensor(IMGS)
            if torch.cuda.is_available():
                IMGS = IMGS.cuda()
            outs = net(IMGS)
            for i in range(len(outs)):
                if torch.argmax(outs[i]).item() == 0:#如果置信度大于0.15则判定为没有感染
                    writer.writerow({'filename':imgname[i],'class':1})
                else:
                    writer.writerow({'filename': imgname[i], 'class': 0})
            IMGS = []
            imgname = []

if __name__ == '__main__':
    filepath = input('输入测试路径名') # filepath为测试路径名（该路径名下有很多图片）
    Result(filepath)