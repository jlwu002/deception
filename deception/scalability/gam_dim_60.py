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
import pickle
from datetime import datetime

np.random.seed(100)
cuda = torch.device('cuda')

def loss_fun(fx, fy, fz,exploits,div_num,dim_single,dim,num_x,version_num):
    #if y = 0, mask; if y = 1, nonmask
    c = 0.01*(dim - torch.sum(fy,1))
    x_2 = fx.repeat_interleave(exploits.size(0), dim = 0)
    x_2_expand = x_2.reshape(num_x*div_num*exploits.size(0),dim_single)
    x_2_expand_sub = x_2_expand[:,:int(dim_single/2)]
    v = torch.sum((x_2_expand_sub!=-1),1) 
    exploits_2 = exploits.repeat(num_x,1,1)
    exploits_2_expand = exploits_2.repeat_interleave(div_num, dim = 0)
    x_2_expand_all = torch.repeat_interleave(x_2_expand,version_num+1).reshape(exploits_2_expand.size())
    delta = torch.eq(x_2_expand_all,exploits_2_expand).sum(2)
    delta = torch.all(delta==1,dim = 1)
    z_2 = torch.flatten(fz)
    z_2_expand = z_2.repeat_interleave(div_num, dim = 0)
    loss = torch.sum(c) + torch.sum(z_2_expand*v*delta)
    loss = loss/(num_x)
    return loss

def loss_fun_z(fx, fy, fz,exploits,div_num,dim_single,dim,num_x,version_num):
    x_2 = fx.repeat_interleave(exploits.size(0), dim = 0)
    x_2_expand = x_2.reshape(num_x*div_num*exploits.size(0),dim_single)
    x_2_expand_sub = x_2_expand[:,:int(dim_single/2)]
    v = torch.sum((x_2_expand_sub!=-1),1) 
    exploits_2 = exploits.repeat(num_x,1,1)
    exploits_2_expand = exploits_2.repeat_interleave(div_num, dim = 0)
    x_2_expand_all = torch.repeat_interleave(x_2_expand,version_num+1).reshape(exploits_2_expand.size())
    delta = torch.eq(x_2_expand_all,exploits_2_expand).sum(2)
    delta = torch.all(delta==1,dim = 1)
    z_2 = torch.flatten(fz)
    z_2_expand = z_2.repeat_interleave(div_num, dim = 0)
    loss = torch.sum(z_2_expand*v*delta)
    loss = loss/(num_x)
    return loss


# dim_single = 20 #has to be an even number


def calculate_loss(dim_single,div_num,num_sim):
    version_num = 3

    tic = time.perf_counter()
    
    num_x = num_sim
    num_r = num_sim
    dim = dim_single*div_num
    
    f = open('exploit_'+str(dim_single)+'_bm.pckl', 'rb')
    exploit_list_test = pickle.load(f)
    f.close()
    exploits = torch.FloatTensor(exploit_list_test).cuda()
    
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

    

    for k in range(1000):        
        if k%100==0:
            mydate = datetime.now()
            csvstr = datetime.strftime(mydate, '%Y, %m, %d, %H, %M, %S')
            text_file = open("bm_"+str(dim_single)+"_"+str(k)+".txt", "w")
            n = text_file.write(csvstr)
            text_file.close()


        nnq = NNQ().cuda()
        nnz = NNZ().cuda()
        
        optimizer_z = torch.optim.Adam(nnz.parameters(), lr=0.01*0.15, betas=(0.5, 0.999))
        optimizer_q = torch.optim.Adam(nnq.parameters(), lr=0.01, betas=(0.5, 0.999))
        r = Variable(torch.rand(num_r,dim).cuda())
        x_rand_all = []
        for ii in range(div_num):
    
            #generate ports
            #-1 if closed, 1 if open  
            x_rand_port = np.random.randint(0, 2, (num_x, int(dim_single/2)))
            
            #check if all ports are closed    
            x_rand_port_check = np.sum(x_rand_port,1)
            resample_idx = (x_rand_port_check==0).nonzero()[0]
            resample_flag = len(resample_idx)!=0
            
            #resample the all ports are closed
            while(resample_flag):
                for idx in resample_idx:
                    x_rand_port[idx] = np.random.randint(0, 2, (1, int(dim_single/2)))
                x_rand_port_check = np.sum(x_rand_port,1)
                resample_idx = (x_rand_port_check==0).nonzero()[0]
                resample_flag = len(resample_idx)!=0
                
            x_rand_port = np.where(x_rand_port == 0, -1, x_rand_port)
            
            #generate system
            #-1 if not installed, otherwise version 1-3
            x_rand_sys_idx = np.random.randint(0, 3, num_x)
            x_rand_sys = -np.ones((num_x, 3))
            
            for idx in range(num_x):
                x_rand_sys[idx][x_rand_sys_idx[idx]]=np.random.randint(1, version_num + 1, 1)[0]
                
            #generate application
            app_dim = int(dim_single/2) - 3
            x_rand_app = np.random.randint(0, version_num + 1, (num_x, app_dim))
            x_rand_app = np.where(x_rand_app == 0, -1, x_rand_app)
        
            x_rand = np.concatenate((x_rand_sys,x_rand_app,x_rand_port), axis = 1)
            
            if len(x_rand_all) == 0:
                x_rand_all = x_rand
            else:
                x_rand_all = np.concatenate((x_rand_all,x_rand), axis = 1)
    
            
            
        x = Variable(torch.FloatTensor(x_rand_all).cuda())
        t = Variable(torch.Tensor([0.5]).cuda())
        #y = (y>t).float() *1
        
        for i in range(501):   
            #Z neural network update
            for epoch in range(4):
                y = nnq(x,r)
                if i%100==0:
                    y = (y>t).float() *1
                yy=y*x
                optimizer_z.zero_grad()
                z = nnz(yy)
                z_loss = -loss_fun_z(x,y,z,exploits,div_num,dim_single,dim,num_x,version_num)
                #loss_all_z = loss_all_z + [-z_loss.item()]
                z_loss.backward()
                optimizer_z.step()
            
            
            #q neural network update
            optimizer_q.zero_grad()
            y = nnq(x,r)
            if i%100==0:
                y = (y>t).float() *1
            yy = y*x
            z = nnz(yy)
            q_loss = loss_fun(x,y,z,exploits,div_num,dim_single,dim,num_x,version_num) 
            #if (i%100==0):
             #   print(q_loss)
                # print(torch.min(y))
                # print(torch.max(y))
           # loss_all_q = loss_all_q + [q_loss.item()]
            q_loss.backward()
            optimizer_q.step()
            
        loss_list_q = loss_list_q + [q_loss.item()]
        loss_list_z = loss_list_z + [-z_loss.item()]

    
    toc = time.perf_counter()
    time_count = toc-tic

    return time_count,loss_list_q,loss_list_z


dim_single = 60
div_num = 1
num_sim = 10000

print("Running...")
time_count,loss_list_q,loss_list_z = calculate_loss(dim_single,div_num,num_sim)
print(time_count)

np.savetxt("bm_orig_loss_q_dim"+str(dim_single)+".csv", loss_list_q, delimiter =", ", fmt ='% s')
np.savetxt("bm_orig_loss_z_dim"+str(dim_single)+".csv", loss_list_z, delimiter =", ", fmt ='% s')

np.savetxt("bm_orig_timecount_dim"+str(dim_single)+".csv", [time_count], delimiter =", ", fmt ='% s')

