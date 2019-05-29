# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:20:20 2018

@author: sunyh
"""

import numpy as np


def load_weight(weights_path):
    
    '''导入训练好的Softmax模型
    input: weights_path 权重存储的位置
    output: weights(mat) 将权重存为矩阵
            m(int),n(int) 权重的行数与列数
    '''
    
    f = open(weights_path)
    w = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights,m,n
    
def load_data(inputfile):
    
    '''
    input: inputfile(string) 训练样本位置
    output：feature_data(mat) 特征
#            label_data(mat) 标签
#            k(int) 类别个数
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
#    return np.mat(feature_data)
    return np.mat(feature_data),np.mat(label_data).T   
    
    
def predict(test_data,weights):
    
    '''利用训练好的softmax模型对测试集进行预测
    input： test_data(mat) 测试数据的特征
            weights(mat) 模型权重
    output: h.argmax(axis=1) 所属类别
    '''
    
    h = test_data*weights
#    print (h)
#    print (h.argmax(axis=1))
    return h.argmax(axis=1) #获取某行中最大元素的索引值，即为所属类别
    

def save_result(file_name,result):
    
    '''保存最终的预测结果
    input: file_name(string) 保存结果的文件名
           result(mat) 最终预测结果
    '''
    
    f_result = open(file_name,"w")
    m = np.shape(result)[0]
#    print (result)
    for i in xrange(m):
        f_result.write(str(result[i, 0])+"\n")
    f_result.close()

def accuracy(real,predict):
    real_list = []
    predict_list = []
    real = real.tolist()
    predict = predict.tolist()
    for i in xrange(len(real)):
        real_list.append(int(real[i][0]))
        predict_list.append(int(predict[i][0]))
       
    loss = (np.mat(real_list)-np.mat(predict_list)).tolist()[0]
    right_count = 0
    err_count = 0
    for x in loss:
        if x==0:
            right_count += 1
        else:
            err_count += 1
#    print (right_count,err_count)
    accuracy = float(right_count)/float(right_count + err_count)
#    print (accuracy)
    return accuracy
if __name__ =="__main__":
    #1.导入softmax模型
    print ("---------1.load model-----------")
    w,m,n = load_weight("weights")  #m为样本的维度，n为类别总数
#    print (w,n,m)
    #2.导入测试数据
    print ("--------2.load test_data----------")
    test_data,real_label_data = load_data('data/MNIST_test_label.txt')
    #利用训练好的softmax模型对测试集进行预测
    print ("--------3.get prediction--------")
    predict_result_label = predict(test_data,w)
    #4.计算准确率
    print ("----------4.precession-----------")
    accuracy = accuracy(real_label_data,predict_result_label)
    print (accuracy)
    #4.保存最终的预测结果
    print ("-----------4.save prediction--------")
    save_result("result",predict_result_label)