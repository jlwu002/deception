# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:03:00 2021

@author: Junlin

Example: 3 dimension

lst value:
    
REPLACE 0 HERE WITH -1    
0: [0, 0, 0]
1: [0, 0, 1]
2: [0, 1, 0]
3: [0, 1, 1]
4: [1, 0, 0]
5: [1, 0, 1]
6: [1, 1, 0]
7: [1, 1, 1]

q_y_x
z_e_xy
"""

import cplex
import itertools
import numpy as np
from itertools import product
import pandas as pd 
import time

tic = time.perf_counter()

dim_single = 6
div_num = 1

c = 0.01

dim = dim_single*div_num

exploits = [[-1,1,1,-1,-1,-1],[-1,-1,1,-1,1,-1]]
obj_value_result_q_all = []
obj_value_result_z_all = []

lst_y = [list(i) for i in itertools.product([0, 1], repeat=dim)]
lst_x = [list(i) for i in itertools.product([-1, 1], repeat=dim)]
lst_x_rep = lst_x*len(lst_y)

def attacker_value(x):
    return (sum(x)+dim_single)/2

cost = []
for y in lst_y:
    cost = cost +[c*(dim-sum(y))]*len(lst_x)

#q_y_x
var_q = ["{}_{}_{}".format("q", i, j) for i in range(0, len(lst_y)) for j in range(0, len(lst_x))]

#get all combinations of x and y
y_x_combination = list(product(lst_y, lst_x))
#xdy_unique[0] is the unique value for $x \odot y$ for z
#xdy_unique[1] is the revrese index for the unique combinations
ydx_unique = np.unique([np.multiply(y,x) for y, x in y_x_combination], return_inverse=True, axis = 0)
ydx_val = ydx_unique[0]
ydx_idx = ydx_unique[1]

#randomly generate a strategy
e_rand = np.random.randint(0, len(exploits), len(ydx_val))
strategy_list = [list(e_rand)]

#u*
var_u = ["u"]

#z_e_xy
var_z_all = ["{}_{}_{}".format("z", i, j) for i in range(0, len(exploits)) for j in range(0, len(ydx_val))]

def delta_compare(xe,x):
    return all([xe[i] <= x[i] for i in range(0, dim_single)])

flag = True
iter_check=0
#def cplex_solve_q():
while flag:
    iter_check = iter_check+1
    problem_q = cplex.Cplex()
    problem_q.objective.set_sense(problem_q.objective.sense.minimize)
    var_names = var_q + var_u
    
    objective = list(np.divide(cost, len(lst_x))) +[1.0]
    lower_bounds = [0.0]*len(var_names)
    upper_bounds = [1.0]*len(var_q) +[cplex.infinity]
    
    problem_q.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names)
    
    constraints = []
    rhs = []
    constraint_senses = []
    
    #one constraint correspond to each strategy in list
    for strategy in strategy_list:
        q_coeff = []
        for i in range(len(var_q)):
            #get the index of e correspond to x*y
            exploit_idx_tmp = strategy[ydx_idx[i]]
            x_multi = lst_x_rep[i]
            x_multi_list = [x_multi[ii:ii+dim_single] for ii in range(0,dim,dim_single)]
            value_list = [attacker_value(xi) for xi in x_multi_list]
            delta_list = [delta_compare(exploits[exploit_idx_tmp],xi) for xi in x_multi_list]
            q_coeff = q_coeff + [sum(np.multiply(value_list,delta_list))/len(lst_x)]
        constraints = constraints + [[var_q + var_u, q_coeff+[-1.0]]]    
    
    rhs = rhs + [0.0]*len(strategy_list)
    constraint_senses = constraint_senses + ["L"]*len(strategy_list)
    
    #constraint: sum of q(y,x) = 1 for any x 
    for ind_x in range(0,len(lst_x)):
        constraints_var_tmp = ["{}_{}_{}".format("q", i , ind_x) for i in range(0, len(lst_y))]
        constraints_val_tmp = [1.0] * len(constraints_var_tmp)
        constraints = constraints + [[constraints_var_tmp, constraints_val_tmp]]
    
    rhs = rhs + [1.0]*len(lst_x)
    constraint_senses = constraint_senses + ["E"]*len(lst_x)
    
    
    problem_q.linear_constraints.add(lin_expr = constraints,
                                   senses = constraint_senses,
                                   rhs = rhs)
    
    problem_q.solve()
    sol_vals = problem_q.solution.get_values()
    obj_value_result_q = problem_q.solution.get_objective_value()
    print("fix z and solve q:")
    print(obj_value_result_q)
    print("##################")
    
    
    obj_value_result_q_all = obj_value_result_q_all + [obj_value_result_q]
    
    df = pd.DataFrame(list(zip(var_names, sol_vals)), 
                   columns =['Var', 'Value']) 
    
    #get q(y,x) distribution
    q_dist_list = sol_vals[:-1]
    
    #def cplex_solve_z(q_dist_list):
    problem_z = cplex.Cplex()
    problem_z.objective.set_sense(problem_z.objective.sense.maximize)
    
    var_names = var_z_all
    var_types = ["C"]*len(var_z_all)
    
    lower_bounds = [0.0]*len(var_names)
    upper_bounds = [1.0]*len(var_z_all) 
    
    #convert the first constraint to obj value -> maximize the problem
    z_coeff = []
    for e_idx in range(0,len(exploits)): #for each exploit
        for k in range(0,len(ydx_val)): #for each ydx, k is the index
            ydx_idx_list = list(np.argwhere(ydx_idx==k).flatten())
            coeff_tmp = 0
            for item in ydx_idx_list:
                x_multi = lst_x_rep[item]
                x_multi_list = [x_multi[ii:ii+dim_single] for ii in range(0,dim,dim_single)]
                delta_list = [delta_compare(exploits[e_idx],xi) for xi in x_multi_list]
                value_list = [attacker_value(xi) for xi in x_multi_list]
                coeff_tmp = coeff_tmp + (q_dist_list[item]*sum(np.multiply(value_list,delta_list))/len(lst_x))
            z_coeff = z_coeff + [coeff_tmp]
 
    
    problem_z.variables.add(obj = z_coeff,
                            lb = lower_bounds,
                            ub = upper_bounds,
                            names = var_names,
                            types = var_types)
       
    constraints = []
    rhs = []
    constraint_senses = []
    
    #constraints: sum z = 1
    for ind_xy in range(0,len(ydx_val)):
        constraints_var_tmp = ["{}_{}_{}".format("z", i , ind_xy) for i in range(0, len(exploits))]
        constraints_val_tmp = [1.0] * len(constraints_var_tmp)
        constraints = constraints + [[constraints_var_tmp, constraints_val_tmp]]
    
    rhs = rhs + [1.0]*len(ydx_val)
    constraint_senses = constraint_senses + ["E"]*len(ydx_val)
    
    problem_z.linear_constraints.add(lin_expr = constraints,
                                   senses = constraint_senses,
                                   rhs = rhs)
    
    problem_z.solve()
    sol_vals = problem_z.solution.get_values()
    obj_value_result_z = problem_z.solution.get_objective_value()
    obj_value_result_z_def = obj_value_result_z +(np.sum(np.multiply(q_dist_list,cost))/len(lst_x))
    print("fix q and solve z:")
    print(obj_value_result_z_def)
    print("##################")
    
    obj_value_result_z_all = obj_value_result_z_all + [obj_value_result_z_def]


    df = pd.DataFrame(list(zip(var_names, sol_vals)), 
                   columns =['Var', 'Value']) 
    
    df2 = df[:-1]
    #get new strategy: z = 1 that correspond to each xy observation
    #add this new strategy to the (active) strategy list
    z_add = df2[df2.Value==1].Var.to_list()
    new_stragety = [0]*len(ydx_val)
    for item in z_add:
        idx_split = item.split("_")
        e_idx_tmp = int(idx_split[1])
        ydx_idx_tmp = int(idx_split[2])
        new_stragety[ydx_idx_tmp] = e_idx_tmp
  
    strategy_list = strategy_list + [new_stragety]


    flag = (abs(obj_value_result_z_def - obj_value_result_q) > 0.00001)
    
    
toc = time.perf_counter()
time_count = toc-tic
print(time_count)
pd_data = {'z': [obj_value_result_z], 'q': [obj_value_result_q], 'time':[time_count]}
df = pd.DataFrame(data=pd_data)
#df.to_csv(str(dim_single)+'_dev'+str(div_num)+'_'+str(c)+'_'+'result.csv')    
    
