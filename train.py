from Net import VGG_16
from sample import Sample_train
from sample import Sample_test
from torch import optim
from torch.autograd import Variable
import torch
from torch import nn
from torch.backends import cudnn


if __name__ == '__main__':
    cudnn.benchmark = True#增加程序的运行效率
    sample = Sample_train()#创建一个采样器
    net = VGG_16()#创建一个网络
    if torch.cuda.is_available():
        net = net.cuda()
    error = nn.CrossEntropyLoss()#定义损失
    optimzer = optim.Adam(lr=0.00001,params=net.parameters())#定义优化器
    #开始训练
    for epoch in range(50000):
        data,realclass = sample.get_batch(10)#采样获得数据和标签
        data = torch.Tensor(data)
        realclass = torch.Tensor(realclass)
        realclass = realclass.long()
        #如果支持cuda就转换为cuda类型
        if torch.cuda.is_available():
            data = data.cuda()
            realclass = realclass.cuda()
        else:
            data = Variable(data)
            realclass = Variable(realclass)
        predictclass = net(data)#将数据传到网络里边获得输出
        train_loss = error(predictclass,realclass)#得到训练损失
        print(train_loss)
        optimzer.zero_grad()#清零梯度
        train_loss.backward()#将误差反向传播
        optimzer.step()#对误差进行优化
        if epoch>=1000 and epoch % 10 == 0:#如果训练了10次就做一下测试看看效果
            sample_test = Sample_test()
            data_test,test_class = sample_test.get_batch(2)
            data_test = torch.Tensor(data_test)
            test_class = torch.Tensor(test_class)
            test_class = test_class.long()
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                test_class = test_class.cuda()
            else:
                data_test = Variable(data_test)
                test_class = Variable(test_class)
            test_out = net(data_test)
            test_loss = error(test_out,test_class)
            print('*'*30)
            print('testloss:',test_loss)
            print('real:',test_class,'predict:',test_out)
            print('*'*30)
        if (epoch+1) % 10000 == 0:#每训练1000次保存一次模型
            torch.save(net,'net_soft.pkl')#保存网络
            torch.save(net.state_dict(),'net_soft_params.pkl')#保存网络参数