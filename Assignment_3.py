#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:23:35 2019

@author: shane
"""

import pandas as pd
import numpy as np
import math
from sklearn import model_selection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#task1
def pre_processing():
    diamonds = pd.read_csv("diamonds.csv")
    cut_array = diamonds.cut.unique()
    print("Type of Cuts:", cut_array)
    color_array = diamonds.color.unique()
    print("Type of Colors:", color_array)
    clarity_array = diamonds.clarity.unique()
    print("Type of Clarity:", clarity_array)
    
    #diamonds = diamonds.sample(1000)
    data = diamonds.groupby(["cut", "color", "clarity"]).apply(lambda x: np.array(x[['carat', 'depth', 'table', 'price']])).to_dict()
    
    features = {}
    target = {}
    
    for key, value in data.items():
        arr1 = []
        arr2 = []
    
        if(len(value) > 800):
            print(key)
            print("Num Datapoints:", len(value)) #Num datapoints for combination
            for dp in value:
                arr1.append([dp[0], dp[1], dp[2]])
                arr2.append(dp[3])
                
            features.update({key: arr1})
            target.update({key: arr2})
    
    print("\n******** Features *******")
    print(features)
    print("\n******** Targets *******")
    print(target)
    return features, target

#task2
def num_coefficients_2(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                if i+j==n:
                    t = t+1
    return t

def calculate_model_function2(deg, feature, p):
    result = np.zeros(feature.shape[0])
    k=0
    for n in range(deg+1):
        for i in range(n+1):
            result += p[k]*(feature[:,0]**i)*(feature[:,1]**(n-i))
            k+=1
    return result

def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t

def calculate_model_function3(deg, feature, p):
    result = np.zeros(feature.shape[0])    
    m=0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        result += p[m]*(feature[:,0]**i)*(feature[:,1]**j)*(feature[:,2]**k)
                        m+=1
    return result

#task3
def linearize2(deg, feature, p0):
    f0 = calculate_model_function2(deg, feature, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function2(deg, feature, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
        
    return f0,J

def linearize3(deg, feature, p0):
    f0 = calculate_model_function3(deg, feature, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function3(deg, feature, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
        
    return f0,J
   
#task4
def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp

#task5
def regression(deg, feature, target):
    max_iter = 5
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        f0,J = linearize3(deg,feature, p0)
        dp = calculate_update(target,f0,J)
        p0 += dp

        test_target = calculate_model_function3(deg,feature, p0)
    return(test_target)
        
#task6
def modelSelection(data, target):
    deg_diff = {}
    arr = np.array([])
    maxK = 5
    kf = model_selection.KFold(n_splits=len(target), shuffle=True)
    for k in range(1,maxK):
        for train_index,test_index in kf.split(data):
            for deg in range(4):
                prediction = regression(deg, data[test_index], target[test_index])
                
                arr = np.append(arr, abs(target[test_index]-prediction))
                deg_diff.update({deg: arr})
            
    min_val = 0
    for key, value in deg_diff.items():
        print("Mean Diff Deg ", key, ":", np.mean(value))
        if(min_val == 0):
            min_val = np.mean(value)
            deg = key
        elif(np.mean(value) < min_val):
            min_val = np.mean(value)
            deg = key
          
    print("The optimal degree is:", deg)
    return deg

#task7
def calculate_covariance(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    sigma0_squared = np.matmul(r.T,r)/(J.shape[0]-J.shape[1])
    cov = sigma0_squared * np.linalg.inv(N)
    return cov

def parameter_estimation(feature, target):
    
    max_iter = 5
    deg = modelSelection(feature, target)
    p0 = np.zeros(num_coefficients_2(deg))
    for i in range(max_iter):
        f0,J = linearize2(deg, feature, p0)
        dp = calculate_update(target,f0,J)
        p0 += dp

    cov = calculate_covariance(target,f0,J)

    print("\n******** Price Estimation *******")
    for i in range(len(p0)):
        print("p["+str(i)+"]=",p0[i]," ±",math.sqrt(cov[i,i]))
    print()
    
    x, y = np.meshgrid(np.arange(np.min(feature[:,0]), np.max(feature[:,0]), 0.1),
        np.arange(np.min(feature[:,1]), np.max(feature[:,1]), 0.1))
    test_data = np.array([x.flatten(), y.flatten()]).transpose()
    test_target = calculate_model_function2(deg,test_data, p0)
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature[:,0],feature[:,1],target,c='r')
    ax.plot_surface(x,y,test_target.reshape(x.shape))
    
    ax.view_init(30, 285)
    plt.show()

def main():
    features, target = pre_processing()
    for key, value in features.items():
        print("\n********", key, "*********")
        parameter_estimation(np.array(value), np.array(target[key]))
    
main()