# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:48:10 2021

@author: Junlin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cuda = torch.device('cuda')


tic = time.perf_counter()

dim_single = 6
div_num = 1
cost_param = 0.01
dim = dim_single*div_num

num_x = 5000
num_r = 5000

exploits = torch.FloatTensor([[-1,1,1,-1,-1,-1],[-1,-1,1,-1,1,-1]]).cuda()

loss_list_q = []
loss_list_z = []
 
class NNQ(nn.Module):
    def __init__(self):
        super(NNQ, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),            
        )
        
        self.model2 = nn.Sigmoid()
         
    def forward(self, x, r):
        x_r = torch.cat((x,r),1)
        y = self.model(x_r)
        y = self.model2(y)
        return y
    
class NNZ(nn.Module):
    def __init__(self):
        super(NNZ, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim,exploits.size(0)),
            nn.ReLU(),
            nn.Linear(exploits.size(0),exploits.size(0)),            
            nn.Softmax(dim = 1)                            
        )
        
    def forward(self,y):
        z = self.model(y)
        return z
    
def loss_fun(fx, fy, fz):
    c = cost_param*(dim - torch.sum(fy,1))
    x_2 = fx.repeat_interleave(exploits.size(0), dim = 0)
    x_2_expand = x_2.reshape(num_x*div_num*exploits.size(0),dim_single)
    v = torch.sum(x_2_expand+1,1)/2    
    exploits_2 = exploits.repeat(num_x,1)
    exploits_2_expand = exploits_2.repeat_interleave(div_num, dim = 0)
    delta = torch.all(torch.le(exploits_2_expand,x_2_expand), dim = 1)
    z_2 = torch.flatten(fz)
    z_2_expand = z_2.repeat_interleave(div_num, dim = 0)
    loss = torch.sum(c) + torch.sum(z_2_expand*v*delta)
    loss = loss/(num_x)
    return loss

def loss_fun_z(fx, fy, fz):
    x_2 = fx.repeat_interleave(exploits.size(0), dim = 0)
    x_2_expand = x_2.reshape(num_x*div_num*exploits.size(0),dim_single)
    v = torch.sum(x_2_expand+1,1)/2    
    exploits_2 = exploits.repeat(num_x,1)
    exploits_2_expand = exploits_2.repeat_interleave(div_num, dim = 0)
    delta = torch.all(torch.le(exploits_2_expand,x_2_expand), dim = 1)
    z_2 = torch.flatten(fz)
    z_2_expand = z_2.repeat_interleave(div_num, dim = 0)
    loss = torch.sum(z_2_expand*v*delta)
    loss = loss/(num_x)
    return loss

for k in range(100):
    nnq = NNQ().cuda()
    nnz = NNZ().cuda()
    
    optimizer_z = torch.optim.Adam(nnz.parameters(), lr=0.04, betas=(0.5, 0.999))
    optimizer_q = torch.optim.Adam(nnq.parameters(), lr=0.01, betas=(0.5, 0.999))
    r = Variable(torch.FloatTensor(np.random.uniform(0, 1, (num_r, dim))).cuda())
    x_rand = np.random.randint(0, 2, (num_x, dim))
    x_rand = np.where(x_rand == 0, -1, x_rand)
    x = Variable(torch.FloatTensor(x_rand).cuda())
    t = Variable(torch.Tensor([0.5]).cuda())
    
    for i in range(101):   
        #Z neural network update
        for epoch in range(1):
            optimizer_z.zero_grad()
            y = nnq(x,r)
            if i%100==0:
                y = (y>t).float() *1
            yy=y*x
            z = nnz(yy)
            z_loss = -loss_fun_z(x,y,z)
            z_loss.backward()
            optimizer_z.step()
        
        #q neural network update
        optimizer_q.zero_grad()
        y = nnq(x,r)
        if i%100==0:
            y = (y>t).float() *1
        yy = y*x
        z = nnz(yy)
        q_loss = loss_fun(x,y,z) 
        q_loss.backward()
        optimizer_q.step()
        
        
    # plt.plot(np.array(loss_all_q))
    # plt.plot(np.array(loss_all_z))
    # plt.title("Neural Network Convergence Plot")
    # plt.ylabel("Objective function value")
    # plt.xlabel("Iteration number")
    # plt.show()
    #loss_all_z_list = loss_all_z_list + [loss_all_z]
    #loss_all_q_list = loss_all_q_list + [loss_all_q]
    
    loss_list_q = loss_list_q + [q_loss.item()]
    loss_list_z = loss_list_z + [z_loss.item()]

np.savetxt(str(dim_single) +'_'+'dev'+str(div_num) + '_'+str(cost_param) + '_q_GAN.csv', loss_list_q, delimiter =", ", fmt ='% s')
np.savetxt(str(dim_single) +'_'+'dev'+str(div_num) + '_'+str(cost_param) + '_z_GAN.csv', loss_list_z, delimiter =", ", fmt ='% s')

toc = time.perf_counter()
time_count = toc-tic
print(time_count)
np.savetxt(str(dim_single) +'_'+'dev'+str(div_num) + '_'+str(cost_param) + '_time_GAN.csv', [time_count], delimiter =", ", fmt ='% s')

print(np.mean(loss_list_q))
print(np.std(loss_list_q))
print(time_count/100)
