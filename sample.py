import numpy as np
import os
import random
import cv2

#采样训练集
class Sample_train:
    def get_batch(self,n):
        x = []
        y = []
        imgs = os.listdir('traindata')
        for i in range(n):
            index = random.randint(0,len(imgs)-1)
            img = imgs[index]
            x.append(cv2.imread('traindata\{}'.format(img)))
            label = int(img.split('.')[1])
            y.append(label)
        x1 = np.array(x)
        x1 = (x1/255-0.5)*2
        x1 = np.transpose(x1,[0,3,1,2])
        return x1,np.array(y)

#采样训练集
# class Sample_train:
#     def get_batch(self,n):
#         x = []
#         y = []
#         imgs = os.listdir('traindata')
#         for i in range(n):
#             y1 = np.zeros([2])
#             index = random.randint(0,len(imgs)-1)
#             img = imgs[index]
#             x.append(cv2.imread('traindata\{}'.format(img)))
#             label = int(img.split('.')[1])
#             y1[label] = 1
#             y.append(y1)
#         x1 = np.array(x)
#         x1 = (x1/255-0.5)*2
#         x1 = np.transpose(x1,[0,3,1,2])
#         y = np.array(y)
#         # y = np.reshape(y,[n,2])
#         return x1,y

#采样测试集
class Sample_test:
    def get_batch(self,n):
        self.x = []
        self.y = []
        imgs = os.listdir('testdata')
        for i in range(n):
            index = random.randint(0,len(imgs)-1)
            img = imgs[index]
            self.x.append(cv2.imread('testdata\{}'.format(img)))
            label = int(img.split('.')[1])
            self.y.append(label)
        self.x1 = np.array(self.x)
        self.x1 = (self.x1/255-0.5)*2
        self.x1 = np.transpose(self.x1, [0, 3, 1, 2])
        return self.x1,np.array(self.y)

#采样测试集
# class Sample_test:
#     def get_batch(self,n):
#         self.x = []
#         self.y = []
#         imgs = os.listdir('testdata')
#         for i in range(n):
#             y1 = np.zeros([2])
#             index = random.randint(0,len(imgs)-1)
#             img = imgs[index]
#             self.x.append(cv2.imread('testdata\{}'.format(img)))
#             label = int(img.split('.')[1])
#             y1[label] = 1
#             self.y.append(y1)
#         self.x1 = np.array(self.x)
#         self.x1 = (self.x1/255-0.5)*2
#         self.x1 = np.transpose(self.x1, [0, 3, 1, 2])
#         return self.x1,np.array(self.y)