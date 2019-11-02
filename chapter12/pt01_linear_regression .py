# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd

df=pd.read_excel(r"/home/pc/桌面/上证指数数据.xlsx")
df1=df.iloc[:100,3:6].values
xtrain_features=torch.FloatTensor(df1)
df2=df.iloc[1:101,7].values
xtrain_labels=torch.FloatTensor(df2)


xtrain=torch.unsqueeze(xtrain_features,dim=1)

ytrain=torch.unsqueeze(xtrain_labels,dim=1)

x, y = torch.autograd.Variable(xtrain), Variable(ytrain)


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x
model = Net(n_feature=4, n_hidden=10, n_output=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
num_epochs = 100000
for epoch in range(num_epochs):
    inputs =x
    target =y
    out = model(inputs) # 前向传播
    loss = criterion(out, target) # 计算loss
    # backward
    optimizer.zero_grad() # 梯度归零
    loss.backward() # 方向传播
    optimizer.step() # 更新参数

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,num_epochs,loss.data[0]))

model.eval()
predict = model(x)
predict = predict.data.numpy()
print(predict)
