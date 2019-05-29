# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:58:18 2018

@author: sunyh
"""

import numpy as np



def load_data(inputfile):
    
    '''
    input: inputfile(string) 训练样本位置
    output：feature_data(mat) 特征
            label_data(mat) 标签
            k(int) 类别个数
    '''
    
    f = open(inputfile) 
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp =  []
        feature_tmp.append(1)  #偏置对应系数
        lines = line.strip().split("\t")
        for i in xrange(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_data.append(lines[-1]) 
        feature_data.append(feature_tmp)  #[1,x1,x2]
    f.close()
    return np.mat(feature_data),np.mat(label_data).T,len(set(label_data))
    

def cost(err,label_data):
    
    '''计算损失函数值
    input： err(mat) exp值
            label_data(mat) 标签值
    output: sum_cost/m (float) 损失函数值
    '''
 
    m = np.shape(err)[0]  #m(样本的个数)
    sum_cost = 0.0
    for i in xrange(m):
        if err[i, label_data[i, 0]]/np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost  -= 0
    return sum_cost / m
    

def gradientAscent(feature_data,label_data,k,maxCycle,alpha):
    
    '''利用梯队下降法训练Softmax模型
    input: feature_data(mat) 特征
           label_data(mat) 标签
           k 类别数
           maxCycle 最大迭代次数
           alpha 学习率          
    output：weights(mat) 权重
    '''
    
    m ,n = np.shape(feature_data)  #m是样本的个数，n是每个样本的维度+1(b)
    weights = np.mat(np.ones((n,k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data*weights) #m*k维
#        print ('err:',err)
        if i % 100 == 0:
            print ("\t-------iter:", i ,\
            ",cost:",cost(err,label_data))
        rowsum = -err.sum(axis=1)  #把err的每行的所有元素加起来
        rowsum = rowsum.repeat(k, axis=1)  #重复写k次作为矩阵的元素，目的是计算err/rowsum
        err = err/rowsum
#        print (err)
        for j in range(m):
            err[j, label_data[j, 0]] += 1
        weights = weights + (alpha/m)*feature_data.T*err
        i += 1
    print (weights)
    return weights


def save_model(wfile_name,weights):
    
    '''保存最终模型
    input： wfile_name（string） 保存的位置
            weights(mat)   softmax模型
    '''
    
    fw = open(wfile_name,"w")
    m, n = np.shape(weights)
#    print (m,n)
    for i in xrange(m):
        w_tmp = []
        for j in xrange(n):
            w_tmp.append(str(weights[i, j]))
        fw.write("\t".join(w_tmp)+"\n")
    fw.close()
    

if __name__ == "__main__":
    inputfile = "./data/MNIST_Training.txt"
    #1.导入训练数据
    print "-------1.load data --------"
    feature,label,k = load_data(inputfile)
    #2.训练Softmax模型
    print "---------2.training----------"
    weights =  gradientAscent(feature,label,k,10000,0.4)
    #保存模型
    print "--------3.save model----------"
    save_model("weights",weights)