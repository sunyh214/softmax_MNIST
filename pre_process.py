# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:05:03 2018

@author: sunyh
"""
import os
import numpy as np
def data_process():
    
    train_data = []
    train_label_list = []
    relative_path = os.getcwd()
    file_name_list = os.listdir(relative_path)
    del file_name_list[-1]
    for i in range(len(file_name_list)):
        train_label_list.append(int(file_name_list[i][0]))
        f = open(file_name_list[i])
        each_dight = [line.strip() for line in f.readlines()]
        each_dight = ','.join(each_dight).replace(',','')
        each_dight_list = [int(x) for x in each_dight]
        each_dight_list.append(train_label_list[i])
        train_data.append(each_dight_list)
    return np.mat(train_data)
    
train_data = data_process()
def save_model(wfile_name,train_data):
    
    '''保存最终模型
    input： wfile_name（string） 保存的位置
            weights(mat)   softmax模型
    '''
    
    fw = open(wfile_name,"w")
    m, n = np.shape(train_data)
#    print (m,n)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(train_data[i, j]))
        fw.write("\t".join(w_tmp)+"\n")
    fw.close()
save_model('MNIST_Training.txt',train_data)