#!/usr/bin/env python
# coding: utf-8

# 
# # Classfication_Algorithm_for_STM-based_Single_Molecule_Enzymology_Data
# 
# ## a brief introduction
# 
# ### 1.	data preprocessing
# 
# #### 1.1 find out all tunnelling data
# 
# 
# Estimating the time series with a time-related linear equation, checking the R2 (Coefficient of determination) of the regression result, if the R2 >0.9, the data is judged as tunneling data which will be delete.
# 
# #### 1.2 find out all abnormal conductance
# 
# 
# Calculate the proportion of the conductance value between -2.6 to -1.8 in each hovering data. If the proportion exceeds 1/10 of the total duration, the data is determined to be abnormal conductance data which will be delete.
# 
# 
# 
# #### 1.3 Determine the data hovering interval
# 
# 
# a. Using the results of cluster analysis which determine the approximate range of conductance from -2.8 to -5.0 to perform a rough segmentation of the data firstly.
# 
# b. Use a time series full-length time window from the beginning and the end, then slowly reduce the length of the time window, perform regression processing similar to 1.1 on the data which is in the time window, and refine the hovering interval.
# 
# 
# ### 2.	Main Program
# 
# #### 2.1 Filter
# 
# 
# Use Savitzky-Golay convolution smoothing algorithm to smooth the data.
#     
# #### 2.2 Fuzzy mathematical classfication
# 
# For the smoothed data, according to the results of the cluster analysis, a Gaussian function is selected as the membership function of the five states, and the type of data is judged by the degree of membership.
# 	
# #### 2.3 Corrector
# 
# Choose an appropriate threshold as a trimmer for abnormal data jumps, and examine whether the stay value of each state is greater than the threshold: if it is greater than the threshold, it will be retained, and if it is less than the threshold, it will be removed.
# 
# ### 3.	Parameter learning
# 
# All the parameters that need to be determined in 2 are learned using genetic algorithms.
# The learning samples are manually labeled, and the samples to be labeled are selected according to the following principles: For each individual time series, its approximate entropy should be as large as possible or as small as possible; at the same time, all samples are approximate The range of entropy should be as large as possible to ensure that the sample group contains the largest information
# 
# 
# 
# 
# 

# In[1]:


# import packages


import os
import pandas as pd
import numpy as np
import scipy.signal as ss
import math
import matplotlib.pyplot as plt
import copy
import time
from sklearn import linear_model
from sklearn.metrics import r2_score
from sko.GA import GA
from multiprocessing import Pool
from numba import jit 


# In[ ]:


# all the function to set-up



def read_file(name):
    
    # excel data file reading
    
    return np.array(pd.read_excel(name,header=None))


def cut_down(sig,beg=-2.8,end=-5.0):
    
    # 1.3.a 
    
    
    length = sig.shape[0]
    
    beg_index = 0
    
    for i in range(length):
        
        if sig[i,1] <= beg:
            
            beg_index = i
            
            break
        
    end_index = -1
    
    for i in range(length):
        
        if sig[length-i-1,1] > end:
            
            end_index = length-i-1
            
            break
    
    
    return sig[beg_index:end_index,:]
    

def cut_down_(sig,beg=-2.8,end=-5.0):
    
    # 1.3.a
    
    length = sig.shape[0]
    
    beg_index = 0
    
    for i in range(length):
        
        if sig[i,0] <= beg:
            
            beg_index = i
            
            break
        
    end_index = -1
    
    for i in range(length):
        
        if sig[length-i-1,0] > end:
            
            end_index = length-i-1
            
            break
    
    
    return sig[beg_index:end_index,:]
    

    
def task1(i):
    
    # 1.1
    
    try:
        
    
        print(i)
        
        sig = read_file(i)
        
        sitmp = cut_down(sig)[:,1]
        
        plt.plot(sitmp,label='raw data',color='black')
        
        reg = linear_model.LinearRegression()
        
        X = np.linspace(1,sitmp.shape[0],sitmp.shape[0]).copy().reshape(-1, 1)
        
        y_true = Y = sitmp.copy().reshape(-1, 1)    
        
        reg.fit(X,Y)
        
        y_pred = reg.predict(X)
        
        r2 = r2_score(y_true, y_pred)
        
        f = open('../pic_r/' + i + '.txt','w')
    
        f.write(str(r2))
    
        f.close()
        
        print(r2)
        
        plt.plot(y_pred)
        
        plt.savefig('../pic_r/' + i + '.jpg')
        
        plt.show()
        
    except:
        
        plt.show()
        
        

def task2(i):
    
    # 1.2
    
    try:
        
    
        counter = 0
            
        sig = read_file(i)
            
        sitmp = cut_down(sig)[:,1]
        
        
        for j in sitmp:
            
            if -2.6 < j < -1.8:
                
                counter += 1
        
        p = counter/(sitmp.shape[0])
        
        if p > 1/10:
            
            print(i)
            
            f = open('../pic_--/' + i + '.txt','w')
            
            f.write(str(p))
            
            f.close()
            
            print(p)
            
            plt.plot(sitmp)
            
            plt.savefig('../pic_--/' + i + '.jpg')
            
            plt.close()
        
    except:
        
        plt.close()    
    
    
def smooth_1(sig,windows,poly):
    
    #2.1
    
    return ss.savgol_filter(np.copy(sig),windows,poly)


def _Fuzzy_classfication_opt_____(ii,sig,trend_1,trend_2,a1,a2,a3,a4,a5,T=[-3.06,-3.44,-3.81,-4.27,-4.77]): 

    
    # 2.2 + 2.3
    
    
    global tmp
    global X
    global tmps
    global index_
    
    
    
    
    
    X = sig
    
    tmp = np.zeros(sig.shape)
    
    length = sig.shape[0]
    
    a = [a1,a2,a3,a4,a5]
    
    for i in range(length):
        
        tmps = [math.e**(-((sig[i]-T[0])/a[0])**2),

               math.e**(-((sig[i]-T[1])/a[1])**2),

               math.e**(-((sig[i]-T[2])/a[2])**2),

               math.e**(-((sig[i]-T[3])/a[3])**2),

               math.e**(-((sig[i]-T[4])/a[4])**2),]
        
        index_ = tmps.index((max(tmps)))
        
        tmp[i] = T[index_]


    plt.plot(tmp,label='clustered data')
    
    plt.plot(sig,label='filtered data')
    
    global uu
    
    global uu_
    
    global uu_counter
    
    global tf
    
    global cod
    
    
    a = 0
    
    uu = []
    
    uu_counter = []
    
    counter = 0
    
    for i in tmp:
        
        counter += 1
                
        if a==0:
            
            a = i
            
            uu.append(T.index(i)+1)
            
        else:
            
            if i != a:
                
                a = i
                
                uu.append(T.index(i)+1)
                
                uu_counter.append(counter-1)
                
                counter = 1
    
    uu_counter.append(counter)
    
    tf = []
    
    for i in range(len(uu)):
        
        if i == 0:
            
            if uu_counter[i] < trend_2[uu[i]-1]:
                
                tf.append(False)
            else:
                
                tf.append(True)
        
        elif 1<= i <= len(uu)-2:
            
            if uu[i-1] == uu[i+1]:
                
                if uu_counter[i] < trend_1[uu[i]-1]:
                    
                    tf.append(False)
                else:
                    
                    tf.append(True)
            
            else:
                
                if uu_counter[i] < trend_2[uu[i]-1]:
                    
                    tf.append(False)
                else:
                    
                    tf.append(True)
        
        elif  i == len(uu)-1:
            
            if uu_counter[i] < trend_2[uu[i]-1]:
                
                tf.append(False)
            else:
                
                tf.append(True)
    

                
                  

    
    uu_ = copy.copy(uu)
                
    cod = 0
                
    for i in range(len(uu)):
        
        if tf[i] == False:
            
            if i == 0:
                
                for j in range(i,len(uu)):
                    
                    if tf[j] == True:
                        
                        
                        tmp[int(cod):int(cod+uu_counter[i])] = T[int(uu[j]-1)]
                        
                        cod += uu_counter[i]
                        
                        break
            
            elif 1 <= i <= len(uu)-1:
                
                
                tmp[int(cod):int(cod+uu_counter[i])] = tmp[int(cod-1)]
                
                
                cod += uu_counter[i]
        
        else:
            
            cod += uu_counter[i]
        
                    
    plt.plot(tmp,label='fix data',color='green')
            
    
    XX = [0,len(sig)]
        
    a = 0
    
    uu = []
    
    for i in tmp:
        
        if a==0:
            
            a = i
            
            uu.append(T.index(i)+1)
        
        else:
            
            if i != a:
                
                a = i
                
                uu.append(T.index(i)+1)
    
    print(uu)
    
    f = open('../pic_debug_cutt_ftttttttttttttttttttt/'+ii+'.txt','w')
    
    f.write(str(uu))
    
    f.write('\n')
    
    f.write(str(tmp.shape[0]))
    
    f.close()
    
        
    plt.fill_between(XX,[T[0]+0.1,T[0]+0.1],[T[0]-0.1,T[0]-0.1],color='mistyrose',alpha=1)
    
    plt.fill_between(XX,[T[1]+0.1,T[1]+0.1],[T[1]-0.1,T[1]-0.1],color='salmon',alpha=0.75)
    
    plt.fill_between(XX,[T[2]+0.1,T[2]+0.1],[T[2]-0.1,T[2]-0.1],color='tomato',alpha=0.75)
    
    plt.fill_between(XX,[T[3]+0.1,T[3]+0.1],[T[3]-0.1,T[3]-0.1],color='orangered',alpha=0.75)
    
    plt.fill_between(XX,[T[4]+0.1,T[4]+0.1],[T[4]-0.1,T[4]-0.1],color='red',alpha=0.75)
    
    plt.title(str(uu))
    
    plt.legend()
    
    
    return tmp


def task(ii):
    
    # 2
    
    global sitmp
    
    
    
    try:
        
        
        print(ii)
        sig = read_file(ii)
        sitmp_ = cut_down(sig)[:,1]
        
        sitmp = sitmp_.copy()
        
        
        
        def XUANTING(sitmp,R):
            
            #
                            
            length = sitmp.shape[0]
                
            X = np.linspace(1,sitmp.shape[0],sitmp.shape[0])
            
            i = 0
            
            for i in range(0,length-1,100):
                
                reg = linear_model.LinearRegression()
                        
            
                X_ = X[i::].copy().reshape(-1, 1)
                
                
                y_true = Y = sitmp[i::].copy().reshape(-1, 1)
            
            
                reg.fit(X_,Y)
                
                y_pred = reg.predict(X_)
                
                r2 = r2_score(y_true, y_pred)
                
                    
                if (reg.coef_).shape[0] == 1 and reg.coef_[0] <0 and (r2 > R):
                    
                    print('pigu')
                    
                    print(i)
                    
                    print(r2)
                                        
                    break
            
            j = 0
                
            for j in range(0,length-1,100):
                
                reg = linear_model.LinearRegression()
                        
                X_ = X[0:length-j].copy().reshape(-1, 1)
                
                y_true = Y = sitmp[0:length-j].copy().reshape(-1, 1)
            
                reg.fit(X_,Y)
                
                y_pred = reg.predict(X_)
                
                r2 = r2_score(y_true, y_pred)
                
                
                if ((reg.coef_).shape[0] == 1) and (reg.coef_[0] <0) and (r2 > R):
                    
                    print('tou')
                    
                    print(j)
                                    
                    print(r2)
                    
                    
                    break
            
            
            
            if i <= length-j:
                
                                
                return sitmp[0:6000]
                
            else:
                
                
                                
                sitmp = sitmp[length-j:i]
                
                
                return sitmp
        
        
        
        sitmp = XUANTING(sitmp,0.9)
        
        print(sitmp.shape)
        
        
        plt.plot(sitmp,label='cut data',color='black')
        
        
        tmp_ = smooth_1(sitmp,1133,4)
        
        
        _Fuzzy_classfication_opt_____(ii,tmp_,[0,50,300,300,300],[480,480,480,480,1000],60.3762132,63.4945734,60.89502323,70.15135817,81.98178009)
        
        plt.savefig('../pic_debug_cutt_ftttttttttttttttttttt/' + ii + '.jpg')
        
        plt.close()
        
        

    except:
        
        
        plt.close()  


def Fuzzy_classfication_opt_____(ii,sig,trend_1,trend_2,a1,a2,a3,a4,a5,T=[-3.06,-3.44,-3.81,-4.27,-4.77]): 

    
    # 3
    
    
    global tmp
    global X
    global tmps
    global index_
    
    
    
    
    
    X = sig
    
    tmp = np.zeros(sig.shape)
    
    length = sig.shape[0]
    
    a = [a1,a2,a3,a4,a5]
    
    for i in range(length):
        
        tmps = [math.e**(-((sig[i]-T[0])/a[0])**2),

               math.e**(-((sig[i]-T[1])/a[1])**2),

               math.e**(-((sig[i]-T[2])/a[2])**2),

               math.e**(-((sig[i]-T[3])/a[3])**2),

               math.e**(-((sig[i]-T[4])/a[4])**2),]
        
        index_ = tmps.index((max(tmps)))
        
        tmp[i] = T[index_]


    plt.plot(tmp,label='clustered data')
    
    plt.plot(sig,label='filtered data')
    
    global uu
    
    global uu_
    
    global uu_counter
    
    global tf
    
    global cod
    
    
    a = 0
    
    uu = []
    
    uu_counter = []
    
    counter = 0
    
    for i in tmp:
        
        counter += 1
                
        if a==0:
            
            a = i
            
            uu.append(T.index(i)+1)
            
        else:
            
            if i != a:
                
                a = i
                
                uu.append(T.index(i)+1)
                
                uu_counter.append(counter-1)
                
                counter = 1
    
    uu_counter.append(counter)
    
    tf = []
    
    for i in range(len(uu)):
        
        if i == 0:
            
            if uu_counter[i] < trend_2[uu[i]-1]:
                
                tf.append(False)
            else:
                
                tf.append(True)
        
        elif 1<= i <= len(uu)-2:
            
            if uu[i-1] == uu[i+1]:
                
                if uu_counter[i] < trend_1[uu[i]-1]:
                    
                    tf.append(False)
                else:
                    
                    tf.append(True)
            
            else:
                
                if uu_counter[i] < trend_2[uu[i]-1]:
                    
                    tf.append(False)
                else:
                    
                    tf.append(True)
        
        elif  i == len(uu)-1:
            
            if uu_counter[i] < trend_2[uu[i]-1]:
                
                tf.append(False)
            else:
                
                tf.append(True)
    

                
                  

    
    uu_ = copy.copy(uu)
                
    cod = 0
                
    for i in range(len(uu)):
        
        if tf[i] == False:
            
            if i == 0:
                
                for j in range(i,len(uu)):
                    
                    if tf[j] == True:
                        
                        
                        tmp[int(cod):int(cod+uu_counter[i])] = T[int(uu[j]-1)]
                        
                        cod += uu_counter[i]
                        
                        break
            
            elif 1 <= i <= len(uu)-1:
                
                
                tmp[int(cod):int(cod+uu_counter[i])] = tmp[int(cod-1)]
                
                
                cod += uu_counter[i]
        
        else:
            
            cod += uu_counter[i]
        
                    
    plt.plot(tmp,label='fix data',color='green')
            
    
    XX = [0,len(sig)]
        
    a = 0
    
    uu = []
    
    for i in tmp:
        
        if a==0:
            
            a = i
            
            uu.append(T.index(i)+1)
        
        else:
            
            if i != a:
                
                a = i
                
                uu.append(T.index(i)+1)
    
    print(uu)
    
    f = open('../pic_debug_cutt_ftttttttttttttttttttt/'+ii+'.txt','w')
    
    f.write(str(uu))
    
    f.write('\n')
    
    f.write(str(tmp.shape[0]))
    
    f.close()
    
        
#    plt.fill_between(XX,[T[0]+0.1,T[0]+0.1],[T[0]-0.1,T[0]-0.1],color='mistyrose',alpha=1)
    
#    plt.fill_between(XX,[T[1]+0.1,T[1]+0.1],[T[1]-0.1,T[1]-0.1],color='salmon',alpha=0.75)
    
#    plt.fill_between(XX,[T[2]+0.1,T[2]+0.1],[T[2]-0.1,T[2]-0.1],color='tomato',alpha=0.75)
    
#    plt.fill_between(XX,[T[3]+0.1,T[3]+0.1],[T[3]-0.1,T[3]-0.1],color='orangered',alpha=0.75)
    
#    plt.fill_between(XX,[T[4]+0.1,T[4]+0.1],[T[4]-0.1,T[4]-0.1],color='red',alpha=0.75)
    
    plt.title(str(uu))
    
    plt.legend()
    
    
    return tmp        
        


def DD_____(x):
    
    # 3
    
    
    x_ = x.copy()
    
    b = [250,1,0,0,0,0,0,0,0,0,0,0,200,200,200,200,200]
    k = [1000,5,100,100,100,100,100,400,400,400,400,400,1000,1000,1000,1000,1000]
    
    for i in range(len(k)):
        
        x_[i] = x_[i]*(k[i]-b[i]) + b[i]
    
    
    print(x_)
    
    trend_1 = [x_[7],x_[8],x_[9],x_[10],x_[11]]
    trend_2 = [x_[12],x_[13],x_[14],x_[15],x_[16]]
    

    windows = 2*math.floor(x_[0])+1
    poly = math.floor(x_[1])
    
    
    try:
        
        counter = 0
        
        tmps = 0
        
        
        for i in range(len(labels)):
            
            counter += 1
            
            plt.plot(sig_labels[counter-1],label='raw data')
            
            plt.plot(labels[counter-1],label='label data')
                
            sigg = smooth_1(sig_labels[counter-1],windows,poly)
            
            Fuzzy_classfication_opt_____('learning',sigg,trend_1,trend_2,x_[2],x_[3],x_[4],x_[5],x_[6])
    
            
            a = 0
            
            uu = 0
            
            for i in tmp:
                
                if a==0:
                    
                    a = i
                    
                    uu = 1
                
                else:
                    
                    if i != a:
                        
                        a = i
                        
                        uu += 1
            
            a = 0
            
            uu_ = 0
            
            for i in labels[counter-1]:
                
                if a==0:
                    
                    a = i
                    
                    uu_ = 1
                
                else:
                    
                    if i != a:
                        
                        a = i
                        
                        uu_ += 1            
    
            if uu > uu_:
                
            
                tmps += (uu/uu_)*((tmp - labels[counter-1])**2).sum()
            
            
            else:
                
                tmps += (uu_/uu)*((tmp - labels[counter-1])**2).sum()
    
        
            plt.show()
                    
    
        print(tmps)
        
        return tmps
        
    except:
        
        plt.show()
        
        print(counter)
        
        
        return 1/counter*100**100


    
@njit
def _phi(N,U,m,r):
    x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
    C = [
        len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
        for x_i in x
    ]
    
    return (N - m + 1.0) ** (-1) * np.sum(math.log(C))    

@njit
def _maxdist(x_i, x_j):
    
    return max([abs(ua - va) for ua, va in zip(x_i, x_j)])



@njit
def ApEn(U, m, r) -> float:

    N = len(U)

    return abs(_phi(N,U,m + 1,r) - _phi(N,U,m,r))    
        

def opt_____():
    
    

    lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    

    ub = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


    
    ga = GA(func=DD_____, n_dim=17, size_pop=50, max_iter=1000, lb=lb, ub=ub, precision=1e-7,prob_mut=0.1)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', -best_y)
    








# # How To Run?

# In[ ]:


# data preprocessing

os.chdir('New folder')
files = os.listdir()


if "__main__" == __name__:
    
    
    start = time.time()
    pool = Pool()
    tofs = pool.map(task1, files)
    end = time.time()
    t = end - start

if "__main__" == __name__:
    
    
    start = time.time()
    pool = Pool()
    tofs = pool.map(task2, files)
    end = time.time()
    t = end - start

os.chdir('../pic_r')
filefile = os.listdir()

wrong = []
for i in filefile:
    if '.txt' in i:
        f = open(i,'r')
        f = f.readlines()
        if float(f[0]) > 0.90:
            print(i)
            wrong.append(i)

f = open('wrong.txt','w')

for i in wrong:
    f.write(i)
    f.write('\n')

f.close()




# loading data for learning

os.chdir('..')
os.chdir('zhanglabel')
simp1 = np.array(pd.read_excel('sam1.xlsx',header=None))
simp2 = np.array(pd.read_excel('sam2.xlsx',header=None))
simp3 = np.array(pd.read_excel('sam3.xlsx',header=None))
simp4 = np.array(pd.read_excel('sam4.xlsx',header=None))
simp5 = np.array(pd.read_excel('sam5.xlsx',header=None))
simp6 = np.array(pd.read_excel('sam6.xlsx',header=None))

simp11 = np.array(pd.read_excel('sam-1.xlsx',header=None))

sig_labels = [simp1[:,0],simp2[:,0],simp3[:,0],simp4[:,0],simp5[:,0],simp6[:,0],]

labels = [simp1[:,1],simp2[:,1],simp3[:,1],simp4[:,1],simp5[:,1],simp6[:,1],]

for i in range(1,5):
    
    tmp_sim = cut_down_(simp11[:,i*3:i*3+2])
    
    counter = 0
    
    for j in tmp_sim[:,1]:
        
        if not(str(j) == 'nan'):
            
            
            tmp_sim[0:counter,1] = j
            
            break
        
        counter += 1
                


    
    counter = -1
    
    for j in tmp_sim[:,1][-1::-1]:
        
        
        
        if not(str(j) == 'nan'):
            
            tmp_sim[counter::,1] = j
            
            break
        
        counter -= 1
    
    
    sig_labels.append(tmp_sim[:,0])
    
    labels.append(tmp_sim[:,1])
    

simp12 = np.array(pd.read_excel('sam11.xlsx',header=None))

for i in range(1,8):
    
    tmp_sim = cut_down_(simp12[:,i*3:i*3+2])
    
    counter = 0
    
    for j in tmp_sim[:,0]:
        
        if str(j) == 'nan':
            
            
            break
        
        counter += 1
    
    
    
    tmp_sim = tmp_sim[0:counter,:]
    
    
    
    counter = 0
    
    for j in tmp_sim[:,1]:
        
        if not(str(j) == 'nan'):
            
            
            tmp_sim[0:counter,1] = j
            
            break
        
        counter += 1
                


    
    counter = -1
    
    for j in tmp_sim[:,1][-1::-1]:
        
        
        
        if not(str(j) == 'nan'):
            
            tmp_sim[counter::,1] = j
            
            break
        
        counter -= 1    

                    
    
    sig_labels.append(tmp_sim[0:counter,0])
    
    labels.append(tmp_sim[0:counter,1])




TT = [-3.06,-3.44,-3.81,-4.27,-4.77]

for i in labels:
    
    for j in range(i.shape[0]):
        
        i[j] = TT[int(i[j]-1)]
        

# Paramater Learning

opt_____()


# Data Processing
os.chdir('New folder')
files = os.listdir()

for ii in range(len(files)):
    i = files[ii]
    task(i)
    

# then copy files in pic_debug_cutt_ftttttttttttttttttttt to pic_debug_cutt_ftttttttttttttttttttt_p by hand


# data delete

diu = set()

f = open('wrong.txt','r')

f = f.readlines()

for i in f:
    
    diu.add(i.replace('\n','').replace('.txt',''))

        

os.chdir('../pic_-')

files = os.listdir()

for i in files:
    
    if '.jpg' in i:
        
        diu.add(i.replace('.jpg',''))

diu = list(diu)

os.chdir('../pic_debug_cutt_ftttttttttttttttttttt_p')

for i in diu:
    
    try:
        
        os.remove(i + '.jpg')
        os.remove(i + '.txt')
    
    except:
        
        continue


# ## 4.Stat

# In[ ]:


import os
import numpy as np

os.chdir('../pic_debug_cutt_ftttttttttttttttttttt_p')
file = os.listdir()
print(os.getcwd())

data = []

for i in file:
    
    if '.txt' in i:
        
    
        f = open(i,'r')
        f = f.readlines()
        
        if int(f[1]) > 6000:
            
        
            data.append(f[0])
            
            if '2, 1, 3, 4' in f[0]:
                
                
                
                print(i)
                
                print(f[0].replace('\n',''))

data_set = set()

for j in data:
    
    data_set.add(j)

data_set = list(data_set)

counter = np.zeros(len(data_set))

counter_set = []

for i in range(len(data_set)):
    counter_set.append([])


for i in data:
    
    index = data_set.index(i)
    
    counter[index] += 1
    
    counter_set[index].append(i)

ff = open('../pic_debug_cutt_ftttttttttttttttttttt_p.txt','w')

for k in range(len(data_set)):
    
    ff.write(data_set[k])
    ff.write('\t')
    ff.write(str(counter[k]))
    ff.write('\n')

ff.close()



want = [[2,1,3,4],
        [2,1,3],
        [1,3,4],
        [3,4,2],
        [4,2,1],
        [2,1],
        [1,3],
        [3,4],
        [4,2]]

cc = []
pp = []

for i in range(len(want)):
    
    cc.append(0)
    pp.append([])

mm = []

for i in want:
    k = str(i)
    k = k.replace('[','').replace(']','')
    mm.append(k)
    print(k)
    counter = 0
    
    for j in data:
        
        if k in j:
            cc[want.index(i)] += 1
            pp[want.index(i)].append(counter)
        
        counter += 1

