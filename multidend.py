"""
Copyright 2020 Toshitake Asabuki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# coding: UTF-8
from __future__ import division
import numpy as np
import pylab as pl
from numba import jit
import numba
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.matlib
import os
import shutil
import sys
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn.decomposition
from tqdm import tqdm


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
params = {'backend': 'ps',
    'axes.labelsize': 11,
    'text.fontsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.figsize': [10 / 2.54, 6 / 2.54]}

N = 200

alpha = 1
theta= 1
beta = 5
@numba.njit(fastmath=True, nogil=True)
def g(x):
    
    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))
    return ans
    
  
beta_gate1= 5
beta_gate2= 5
beta_gate3= 5
theta_gate1= 1
theta_gate2= 1
theta_gate3= 1
    
@numba.njit( parallel=True,fastmath=True, nogil=True)
def learning(w,V_star,PSP_star,eps,f):
    for i in numba.prange(len(w[:,0])):
        for l in numba.prange(len(w[0,:])):
            delta=(-(g(V_star[i]))+f[i])*PSP_star[l]
            
            w[i,l]+=eps[i]*delta*beta*(1-g(V_star[i]))

    return w
    
@numba.njit( parallel=True,fastmath=True, nogil=True)
def calc_jitter(spike_mat,time,matrix):
    
    for i in numba.prange(time):
        for j in numba.prange(n_in):
            if spike_mat[j,i]==1:
                #spike_mat[j,i]=0
                matrix[j,min(i+int(np.random.normal(0,jitter_dev)),time-1)]=1
    return matrix
    
gating_rec=True #gating


jitter_dev=0
trials=20


width = 40
dt = 1
window = 300

mu = np.zeros(N)
mu_square = np.zeros(N)+1
mu2 = np.zeros(N)
mu_square2 = np.zeros(N)+1
mu3 = np.zeros(N)
mu_square3 = np.zeros(N)+1

mu_gat = np.zeros(N)
mu_square_gat = np.zeros(N)+1
mu_gat2 = np.zeros(N)
mu_square_gat2 = np.zeros(N)+1
mu_gat3 = np.zeros(N)
mu_square_gat3 = np.zeros(N)+1

gamma_mu=0.0001
gamma_mu_square=gamma_mu

nsecs=width*50000
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)

orientation = np.arange(0,180,1)

tau =15
tau_syn = 5
tau_syn_rec = 5
gain=0.05
n_syn = 28*28*2
n_in =  n_syn
PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)

PSP_rec = np.zeros(N)
PSP_unit_rec = np.zeros(N)
I_syn_rec = np.zeros(N)

g_L=1/tau
g_d_max=1
g_d_fixed=0.7
w  = np.zeros((N,n_syn))
w2  = np.zeros((N,n_syn))
w3  = np.zeros((N,n_syn))
w_gat  = np.zeros((N,N))
w_gat2  = np.zeros((N,N))
w_gat3  = np.zeros((N,N))
w_rec  = np.zeros((N,N))

mask_w=np.zeros((N,n_syn))
mask_w2=np.zeros((N,n_syn))
mask_w3=np.zeros((N,n_syn))
mask_w_gat=np.zeros((N,N))
mask_w_gat2=np.zeros((N,N))
mask_w_gat3=np.zeros((N,N))
mask_w_rec=np.zeros((N,N))

p_in=1
p_rec=1
for i in range(N):
    for j in range(n_syn):
        if np.random.rand()<p_in:
            w[i,j]=np.random.randn()/np.sqrt(p_in*n_syn)*1
            mask_w[i,j]=1
            
for i in range(N):
    for j in range(n_syn):
        if np.random.rand()<p_in:
            w2[i,j]=np.random.randn()/np.sqrt(p_in*n_syn)*1
            mask_w2[i,j]=1
            
for i in range(N):
    for j in range(n_syn):
        if np.random.rand()<p_in:
            w3[i,j]=np.random.randn()/np.sqrt(p_in*n_syn)*1
            mask_w3[i,j]=1
            
for i in range(N):
    for j in range(N):
        if np.random.rand()<p_rec:
            w_gat[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
            mask_w_gat[i,j]=1

for i in range(N):
    for j in range(N):
        if np.random.rand()<p_rec:
            w_gat2[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
            mask_w_gat2[i,j]=1
   
for i in range(N):
   for j in range(N):
       if np.random.rand()<p_rec:
           w_gat3[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
           mask_w_gat3[i,j]=1
               
for i in range(N):
    for j in range(N):
        if np.random.rand()<p_rec:
            w_rec[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
            mask_w_rec[i,j]=1
            
poisson_signal =10
poisson_noise = 0.
eps = 10**(-4)
eps2 = 10**(-4)

rate_in=np.zeros(n_in)

p_connect = 1
max_rate = poisson_signal
w_inh_max = 1/np.sqrt(N)

spike_time = np.zeros(N)

w_inh =np.ones((N,N))*w_inh_max

mask = np.zeros((N,N))
for i in range(N):
    for j in range(N):

            if np.random.rand()<p_connect:
                mask[i,j]=1
w_inh*=mask
w_inh[w_inh<0] = 0
w_inh[w_inh>w_inh_max] = w_inh_max

V_dend = np.zeros(N)

V_som = np.zeros(N)

connection_list = np.zeros((N,n_in),dtype=bool)
for i in range(N):
    connection_list[i,np.random.choice(np.arange(n_in), n_syn, replace = False)]=1

f = np.zeros(N)


test_len = width*180
plot_len=test_len

m = 0

V_star=np.zeros(N)
PSP_star=np.zeros((N,n_syn))

start=0


print("")
print("***********")
print("Learning... ")
print("***********")

f_list = np.zeros((N,simtime_len))

count=0
for i in tqdm(range(simtime_len), desc="[training]"):
 

    if i==start:
        start=i+width
        
        label=np.random.randint(0,180)#int(count%180)#np.random.randint(0,180)
        selected_orientation=orientation[label]
        count+=1
        image=np.zeros((28,28))
        x1=1
        y1=np.tan(selected_orientation/180*np.pi)
        #print(y1)
        x2 = 0
        y2=0
        for l in range(28):
            for j in range(28):
                x3=l-14
                y3=j-14
                u = numpy.array([x2 - x1, y2 - y1])
                v = numpy.array([x3 - x1, y3 - y1])
                L = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
                if L < 3.5:
                    image[l,j]=1
                if np.random.rand()<0.1:
                    if image[l,j]==0:
                        image[l,j]=1
                    else:
                        image[l,j]=0
                if (l-14)**2+(j-14)**2>=14**2:
                    image[l,j]=0
    
        input=image.reshape(784)
        rate_in[0:28*28] = input*poisson_signal
        rate_in[28*28:n_in] = input*poisson_signal

    #print(np.max(rate_in))
    prate = dt*rate_in*(10**-3)
    id = (np.random.rand(n_in)<prate)

    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    
    V_dend1 = np.dot(w,PSP_unit)
    V_dend2 = np.dot(w2,PSP_unit)
    V_dend3 = np.dot(w3,PSP_unit)

    mu +=gamma_mu*(V_dend1-mu)
    mu_square += gamma_mu_square*((V_dend1)**2-mu_square)
    V_dend_hat=(V_dend1-mu) / np.sqrt((mu_square-mu**2))
    
    mu2 +=gamma_mu*(V_dend2-mu2)
    mu_square2 += gamma_mu_square*((V_dend2)**2-mu_square2)
    V_dend_hat2=(V_dend2-mu2) / np.sqrt((mu_square2-mu2**2))
    
    mu3 +=gamma_mu*(V_dend3-mu3)
    mu_square3 += gamma_mu_square*((V_dend3)**2-mu_square3)
    V_dend_hat3=(V_dend3-mu3) / np.sqrt((mu_square3-mu3**2))
    
    
    V_gat=np.dot(w_gat,PSP_unit_rec)
    V_gat2=np.dot(w_gat2,PSP_unit_rec)
    V_gat3=np.dot(w_gat3,PSP_unit_rec)

    g_d=np.exp(beta_gate1*(V_gat))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max
    g_d2=np.exp(beta_gate2*(V_gat2))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max
    
    g_d3=np.exp(beta_gate3*(V_gat3))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max

    V_som = (1.0-dt/tau)*V_som +g_d*(V_dend_hat-V_som)+g_d2*(V_dend_hat2-V_som)+g_d3*(V_dend_hat3-V_som)-np.dot(w_inh,PSP_unit_rec)/tau

    V_star = (g_d*V_dend1+g_d2*V_dend2+g_d3*V_dend3)/(g_L+g_d_max)
    #print(g_d+g_d2)

    f=g(V_som)
    f_list[:,i]=f
    id_rec = np.random.rand(N)<f*gain

    if i>=window*width:
        w=learning(w,V_star,PSP_unit,eps*g_d/(g_L+g_d_max),f)
        w2=learning(w2,V_star,PSP_unit,eps*g_d2/(g_L+g_d_max),f)
        w3=learning(w3,V_star,PSP_unit,eps*g_d3/(g_L+g_d_max),f)

        if gating_rec==True:
            #g_d_sum=-(V_dend1*g_d/g_d_max+V_dend2*g_d2/g_d_max+V_dend3*g_d3/g_d_max)#*beta_gate
            g_sum=g_L+g_d_max
            
            
            rate1 = 1/g_sum * beta_gate1*(g_d*(1-g_d/g_d_max)*V_dend1+g_d2*(0-g_d/g_d_max)*V_dend2+g_d3*(0-g_d/g_d_max)*V_dend3)
            rate2 = 1/g_sum * beta_gate2*(g_d*(0-g_d2/g_d_max)*V_dend1+g_d2*(1-g_d2/g_d_max)*V_dend2+g_d3*(0-g_d2/g_d_max)*V_dend3)
            rate3 = 1/g_sum * beta_gate3*(g_d*(0-g_d3/g_d_max)*V_dend1+g_d2*(0-g_d3/g_d_max)*V_dend2+g_d3*(1-g_d3/g_d_max)*V_dend3)
            
            w_gat=learning(w_gat,V_star,PSP_unit_rec, eps2*rate1,f)
            w_gat2=learning(w_gat2,V_star,PSP_unit_rec, eps2*rate2,f)
            w_gat3=learning(w_gat3,V_star,PSP_unit_rec, eps2*rate3,f)
    
    I_syn_rec = (1.0 - dt / tau_syn_rec) * I_syn_rec
    I_syn_rec[id_rec]+=1/tau/tau_syn_rec
    PSP_rec = (1.0 - dt / tau) * PSP_rec + I_syn_rec
    PSP_unit_rec=PSP_rec*25


print("")
print("***********")
print("Testing... ")
print("***********")

V_dend_list =np.zeros((N,test_len))
V_dend_list2 =np.zeros((N,test_len))
V_dend_list3 =np.zeros((N,test_len))
V_gat_list =np.zeros((N,test_len))
V_som_list=np.zeros((N,test_len))

f_list = np.zeros((N,test_len))
g_d_list = np.zeros((N,test_len))


id = np.zeros((test_len,n_in),dtype=bool)
id_rec = np.zeros((test_len,N),dtype=bool)


start0_list=[]
start1_list=[]
start2_list=[]
start3_list=[]
start4_list=[]
start5_list=[]
start6_list=[]
start7_list=[]
start8_list=[]
start9_list=[]
label_list=[]
for loop in range(trials):
    Summed_EPSP = np.zeros((N,test_len))
    Summed_IPSP = np.zeros((N,test_len))

    PSP = np.zeros(n_in)
    I_syn = np.zeros(n_in)
    I_syn_rec=np.zeros(N)
    PSP_rec=np.zeros(N)
    synaptic_input_matrix=np.zeros((n_in*N,test_len))
    V_dend = np.zeros(N)

    V_som = np.zeros(N)


    start=0
    count=0
    for i in range(test_len):


        if i==start:
            start=i+width
            label=count#np.random.choice(np.arange(0,180,18),1)#randint(0,180)
            count+=1
            #print(label)
            if label==0:
                start0_list.append(i)
            if label==18*1:
                start1_list.append(i)
            if label==18*2:
                start2_list.append(i)
            if label==18*3:
                start3_list.append(i)
            if label==18*4:
                start4_list.append(i)
            if label==18*5:
                start5_list.append(i)
            if label==18*6:
                start6_list.append(i)
            if label==18*7:
                start7_list.append(i)
            if label==18*8:
                start8_list.append(i)
            if label==18*9:
                start9_list.append(i)
            selected_orientation=orientation[label]
            #print(selected_orientation)
            image=np.zeros((28,28))
            x1=1
            y1=np.tan(selected_orientation/180*np.pi)
            #print(y1)
            x2 = 0
            y2=0
            for l in range(28):
                for j in range(28):
                    x3=l-14
                    y3=j-14
                    u = numpy.array([x2 - x1, y2 - y1])
                    v = numpy.array([x3 - x1, y3 - y1])
                    L = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
                    if L < 3.5:
                        image[l,j]=1
                    if np.random.rand()<0.1:
                        if image[l,j]==0:
                            image[l,j]=1
                        else:
                            image[l,j]=0
                    if (l-14)**2+(j-14)**2>=14**2:
                        image[l,j]=0
        
            input=image.reshape(784)
            rate_in[0:28*28] = input*poisson_signal
            rate_in[28*28:n_in] = input*poisson_signal
            
            label_list.append(label)

        
        prate = dt*rate_in*(10**-3)
        
        id[i,:] = (np.random.rand(n_in)<prate)

        I_syn = (1.0 - dt / tau_syn) * I_syn
        I_syn[id[i,:]]+=1/tau/tau_syn
        PSP = (1.0 - dt / tau) * PSP + I_syn
        PSP_unit=PSP*25
        
        V_dend1 = np.dot(w,PSP_unit)
        V_dend2 = np.dot(w2,PSP_unit)
        V_dend3 = np.dot(w3,PSP_unit)

        mu +=gamma_mu*(V_dend1-mu)
        mu_square += gamma_mu_square*((V_dend1)**2-mu_square)
        V_dend_hat=(V_dend1-mu) / np.sqrt((mu_square-mu**2))
        
        mu2 +=gamma_mu*(V_dend2-mu2)
        mu_square2 += gamma_mu_square*((V_dend2)**2-mu_square2)
        V_dend_hat2=(V_dend2-mu2) / np.sqrt((mu_square2-mu2**2))
        
        mu3 +=gamma_mu*(V_dend3-mu3)
        mu_square3 += gamma_mu_square*((V_dend3)**2-mu_square3)
        V_dend_hat3=(V_dend3-mu3) / np.sqrt((mu_square3-mu3**2))
        
        
        V_gat=np.dot(w_gat,PSP_unit_rec)
        V_gat2=np.dot(w_gat2,PSP_unit_rec)
        V_gat3=np.dot(w_gat3,PSP_unit_rec)

        g_d=np.exp(beta_gate1*(V_gat))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max
        g_d2=np.exp(beta_gate2*(V_gat2))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max
        
        g_d3=np.exp(beta_gate3*(V_gat3))/(np.exp(beta_gate1*(V_gat))+np.exp(beta_gate2*(V_gat2))+np.exp(beta_gate3*(V_gat3)))*g_d_max

        V_som = (1.0-dt/tau)*V_som +g_d*(V_dend_hat-1*V_som)+g_d2*(V_dend_hat2-1*V_som)+g_d3*(V_dend_hat3-1*V_som)-np.dot(w_inh,PSP_unit_rec)/tau
        
        for k in range(N):
            f[k] = g(V_som[k])
        id_rec[i,:] = np.random.rand(N)<f*gain
        f_list[:,i]+=f/trials
        g_d_list[:,i]+=g_d/trials
        V_dend_list[:,i] += g(V_dend1)/trials
        V_dend_list2[:,i] += g(V_dend2)/trials
        V_dend_list3[:,i] += g(V_dend3)/trials
        V_gat_list[:,i] = V_gat
        V_som_list[:,i]=V_som
        I_syn_rec = (1.0 - dt / tau_syn_rec) * I_syn_rec
        I_syn_rec[id_rec[i,:]]+=1/tau/tau_syn_rec
        PSP_rec = (1.0 - dt / tau) * PSP_rec + I_syn_rec
        PSP_unit_rec=PSP_rec*25

        Summed_EPSP[:,i] = np.dot(w_gat*(w_gat>0),PSP_unit_rec)

        Summed_IPSP[:,i] = np.dot(w_gat*(w_gat<0),PSP_unit_rec)

np.savetxt('V_dend_list_orientation.txt', V_dend_list, delimiter=',')
np.savetxt('V_gat_list_orientation.txt', V_gat_list, delimiter=',')
np.savetxt('V_som_list_orientation.txt', V_som_list, delimiter=',')

activity_mat=np.zeros((N,180))
for i in range(test_len):
    activity_mat[:,int(i/(width))]+=f_list[:,i]/width
    
dend1_mat=np.zeros((N,180))
for i in range(test_len):
    dend1_mat[:,int(i/(width))]+=V_dend_list[:,i]/width
    
dend2_mat=np.zeros((N,180))
for i in range(test_len):
    dend2_mat[:,int(i/(width))]+=V_dend_list2[:,i]/width

dend3_mat=np.zeros((N,180))
for i in range(test_len):
    dend3_mat[:,int(i/(width))]+=V_dend_list3[:,i]/width
    
theta = np.linspace(0,np.pi,180)


for i in range(10):
    plt.close()
    som_vec=activity_mat[i,0:180:1]
    dend1_vec=dend1_mat[i,0:180:1]
    dend2_vec=dend2_mat[i,0:180:1]
    dend3_vec=dend3_mat[i,0:180:1]

    fig = plt.figure(figsize=(10,2))
    axes0 = fig.add_subplot(141, projection='polar')
    axes0.set_thetamin(0)
    axes0.set_thetamax(180)
    axes0.plot(theta,som_vec)
    axes0.set_xlim([0, np.pi])
    axes0.set_xticks(np.linspace(0, np.pi, 3)[1:])
    axes0.set_yticks([])
    axes0.set_xticklabels([])
    axes0.set_yticklabels([])
    plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['+90$^\circ$',"",'0$^\circ$',"",'-90$^\circ$'],fontsize=10)

    axes1 = fig.add_subplot(142, projection='polar')
    axes1.set_thetamin(0)
    axes1.set_thetamax(180)
    axes1.plot(theta,dend1_vec)
    axes1.set_xlim([0, np.pi])
    axes1.set_xticks(np.linspace(0, np.pi, 3)[1:])
    axes1.set_yticks([])
    axes1.set_xticklabels([])
    axes1.set_yticklabels([])
    plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    #plt.xticks([0,np.pi/2,np.pi],['-90$^\circ$','0$^\circ$','+90$^\circ$'],fontsize=10)

    axes2 = fig.add_subplot(143, projection='polar')
    axes2.set_thetamin(0)
    axes2.set_thetamax(180)
    axes2.plot(theta,dend2_vec)
    axes2.set_xlim([0, np.pi])
    axes2.set_xticks(np.linspace(0, np.pi, 3)[1:])
    axes2.set_yticks([])
    axes2.set_xticklabels([])
    axes2.set_yticklabels([])
    plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    #plt.xticks([0,np.pi/2,np.pi],['-90$^\circ$','0$^\circ$','+90$^\circ$'],fontsize=10)

    axes3 = fig.add_subplot(144, projection='polar')
    axes3.set_thetamin(0)
    axes3.set_thetamax(180)
    axes3.plot(theta,dend3_vec)
    axes3.set_xlim([0, np.pi])
    axes3.set_xticks(np.linspace(0, np.pi, 3)[1:])
    axes3.set_yticks([])
    axes3.set_xticklabels([])
    axes3.set_yticklabels([])
    plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    #plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    #plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['+90$^\circ$',"",'0$^\circ$',"",'-90$^\circ$'],fontsize=10)
    #fig.subplots_adjust(left=0.15,bottom=0.001,right=0.9)
    plt.savefig('polar%s.pdf'%int(i), fmt='pdf', dpi=350)
    
sample_len = 180

max1 = np.zeros(N)
min1 = np.zeros(N)
for i in range(N):
    max1[i] = np.max(activity_mat[i,0:sample_len])
    min1[i] = np.min(activity_mat[i,0:sample_len])
avg_norm1 = np.zeros((N,sample_len))

for i in range(N):
    avg_norm1[i,:] = (activity_mat[i,0:sample_len]-min1[i])/(max1[i]-min1[i])

t = np.zeros(N)
for j in range(N):
    arg = np.angle(np.dot(avg_norm1[j,:],np.exp(np.arange(sample_len)/(sample_len)*2*np.pi*1j))/sum(avg_norm1[j,:]))
    if arg<0:
        arg += 2*np.pi
    t[j] = sample_len/(2*np.pi)*arg

index = np.zeros(N)

index = np.argsort(t)
avg_sorted = np.zeros((N,sample_len))
for i in range(N):
    avg_sorted[i,:] = avg_norm1[int(index[i]),:]

fig, ax = plt.subplots(figsize=(3,3))

cax=plt.imshow(avg_sorted, interpolation='nearest', aspect="auto",cmap='jet')

cbar = fig.colorbar(cax, ticks=[0, 1], orientation='vertical')
cbar.ax.set_yticklabels(['0', '1'],fontsize=10)

plt.xlabel("Stimulus",fontsize=10)
plt.ylabel("Neurons (sorted)",fontsize=10)
plt.yticks([0,N-1],["1","%d"%N],fontsize=10)
ax.tick_params(length=1.3, width=0.05, labelsize=10)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
plt.ylim([-0.5,N-0.5])
pl.xlim([0,sample_len])
plt.xticks([0,89,179],['0','90','180'],fontsize=11)
fig.subplots_adjust(left=0.15,bottom=0.25,right=1)
#for l in range(sample_num):
    #ax.axvline(x=width*len(chunk)*(l+1), ymin=0, ymax=N, color='gray', linewidth=1)

plt.savefig('activity_map.pdf', fmt='pdf',dpi=350)
######################################################
orientation = np.arange(0,180,18)
count=1
fig = plt.figure(figsize=(8, 2))
ax = fig.add_subplot(111)
for k in range(10):
    selected_orientation=orientation[k]
    
    image=np.zeros((28,28))
    x1=1
    y1=np.tan(selected_orientation/180*np.pi)
    
    x2 = 0
    y2=0
    for i in range(28):
        for j in range(28):
            x3=i-14
            y3=j-14
            u = numpy.array([x2 - x1, y2 - y1])
            v = numpy.array([x3 - x1, y3 - y1])
            L = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
            if L < 3.5:
                image[i,j]=1
            if np.random.rand()<0.1:
                if image[i,j]==0:
                    image[i,j]=1
                else:
                    image[i,j]=0
            if (i-14)**2+(j-14)**2>=14**2:
                image[i,j]=0

    pl.subplot(1, 10, count)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    count+=1
plt.savefig('stimuli.pdf', fmt='pdf', dpi=350)

weight=np.zeros((28*28,N))
weight+=(w.T[0:28*28,:]+w.T[28*28:2*28*28,:])
weight/=2
#weight[weight<=0]=0
fig = plt.figure(figsize=(8, 2))
ax = fig.add_subplot(111)
count=1
for i in range(10):
    #index2 = np.argmax(avg_sorted[:,i])
    data= weight[:,int(i)]
    pl.subplot(1, 10, count)
    pl.axis('off')
    pl.imshow(data.reshape(28, 28), cmap=pl.cm.gray_r, interpolation='nearest')
    count+=1
plt.savefig('weights.pdf', fmt='pdf', dpi=350)

weight=np.zeros((28*28,N))
weight+=(w2.T[0:28*28,:]+w2.T[28*28:2*28*28,:])
weight/=2
#weight[weight<=0]=0
fig = plt.figure(figsize=(8, 2))
ax = fig.add_subplot(111)
count=1
for i in range(10):
    #index2 = np.argmax(avg_sorted[:,i])
    data= weight[:,int(i)]
    pl.subplot(1, 10, count)
    pl.axis('off')
    pl.imshow(data.reshape(28, 28), cmap=pl.cm.gray_r, interpolation='nearest')
    count+=1
plt.savefig('weights2.pdf', fmt='pdf', dpi=350)

weight=np.zeros((28*28,N))
weight+=(w3.T[0:28*28,:]+w3.T[28*28:2*28*28,:])
weight/=2
#weight[weight<=0]=0
fig = plt.figure(figsize=(8, 2))
ax = fig.add_subplot(111)
count=1
for i in range(10):
    #index2 = np.argmax(avg_sorted[:,i])
    data= weight[:,int(i)]
    pl.subplot(1, 10, count)
    pl.axis('off')
    pl.imshow(data.reshape(28, 28), cmap=pl.cm.gray_r, interpolation='nearest')
    count+=1
plt.savefig('weights3.pdf', fmt='pdf', dpi=350)

fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1)
for i in range(10):
    pl.subplot(10,1,i+1)
    pl.plot(f_list[i,:],lw=1.5)
    for i in (start0_list):
        pl.axvspan(i, i + width, facecolor='r', alpha=0.3,linewidth=0)
    for i in (start1_list):
        pl.axvspan(i, i + width, facecolor='g', alpha=0.3,linewidth=0)
    for i in (start2_list):
        pl.axvspan(i, i + width, facecolor='b', alpha=0.3,linewidth=0)
    for i in (start3_list):
        pl.axvspan(i, i + width, facecolor='y', alpha=0.3,linewidth=0)
    for i in (start4_list):
        pl.axvspan(i, i + width, facecolor='m', alpha=0.3,linewidth=0)
    for i in (start5_list):
        pl.axvspan(i, i + width, facecolor='gray', alpha=0.3,linewidth=0)
    for i in (start6_list):
        pl.axvspan(i, i + width, facecolor='orange', alpha=0.3,linewidth=0)
    for i in (start7_list):
        pl.axvspan(i, i + width, facecolor='cyan', alpha=0.3,linewidth=0)
    for i in (start8_list):
        pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)
    for i in (start9_list):
        pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)
    pl.xlim([0,test_len])
    pl.ylim([-0.1,1.1])
#plt.xlabel("Time [ms]", fontsize=11)
#plt.ylabel("Activity of soma [a.u.]", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.savefig('soma.pdf', fmt='pdf', dpi=350)


fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1)
for i in range(10):
    pl.subplot(10,1,i+1)
    pl.plot(V_dend_list[i,:],lw=1.5)
    for i in (start0_list):
        pl.axvspan(i, i + width, facecolor='r', alpha=0.3,linewidth=0)
    for i in (start1_list):
        pl.axvspan(i, i + width, facecolor='g', alpha=0.3,linewidth=0)
    for i in (start2_list):
        pl.axvspan(i, i + width, facecolor='b', alpha=0.3,linewidth=0)
    for i in (start3_list):
        pl.axvspan(i, i + width, facecolor='y', alpha=0.3,linewidth=0)
    for i in (start4_list):
        pl.axvspan(i, i + width, facecolor='m', alpha=0.3,linewidth=0)
    for i in (start5_list):
        pl.axvspan(i, i + width, facecolor='gray', alpha=0.3,linewidth=0)
    for i in (start6_list):
        pl.axvspan(i, i + width, facecolor='orange', alpha=0.3,linewidth=0)
    for i in (start7_list):
        pl.axvspan(i, i + width, facecolor='cyan', alpha=0.3,linewidth=0)
    for i in (start8_list):
        pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)
    for i in (start9_list):
        pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)
    pl.xlim([0,test_len])
    pl.ylim([-0.1,1.1])
#plt.xlabel("Time [ms]", fontsize=11)
#plt.ylabel("Activity of soma [a.u.]", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.savefig('dend.pdf', fmt='pdf', dpi=350)


fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1)
for i in range(10):
    pl.subplot(10,1,i+1)
    pl.plot(V_dend_list2[i,:],lw=1.5)
    for i in (start0_list):
        pl.axvspan(i, i + width, facecolor='r', alpha=0.3,linewidth=0)
    for i in (start1_list):
        pl.axvspan(i, i + width, facecolor='g', alpha=0.3,linewidth=0)
    for i in (start2_list):
        pl.axvspan(i, i + width, facecolor='b', alpha=0.3,linewidth=0)
    for i in (start3_list):
        pl.axvspan(i, i + width, facecolor='y', alpha=0.3,linewidth=0)
    for i in (start4_list):
        pl.axvspan(i, i + width, facecolor='m', alpha=0.3,linewidth=0)
    for i in (start5_list):
        pl.axvspan(i, i + width, facecolor='gray', alpha=0.3,linewidth=0)
    for i in (start6_list):
        pl.axvspan(i, i + width, facecolor='orange', alpha=0.3,linewidth=0)
    for i in (start7_list):
        pl.axvspan(i, i + width, facecolor='cyan', alpha=0.3,linewidth=0)
    for i in (start8_list):
        pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)
    for i in (start9_list):
        pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)
    pl.xlim([0,test_len])
    pl.ylim([-0.1,1.1])
#plt.xlabel("Time [ms]", fontsize=11)
#plt.ylabel("Activity of soma [a.u.]", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.savefig('dend2.pdf', fmt='pdf', dpi=350)



fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1)
for i in range(10):
    pl.subplot(10,1,i+1)
    pl.plot(V_dend_list3[i,:],lw=1.5)
    for i in (start0_list):
        pl.axvspan(i, i + width, facecolor='r', alpha=0.3,linewidth=0)
    for i in (start1_list):
        pl.axvspan(i, i + width, facecolor='g', alpha=0.3,linewidth=0)
    for i in (start2_list):
        pl.axvspan(i, i + width, facecolor='b', alpha=0.3,linewidth=0)
    for i in (start3_list):
        pl.axvspan(i, i + width, facecolor='y', alpha=0.3,linewidth=0)
    for i in (start4_list):
        pl.axvspan(i, i + width, facecolor='m', alpha=0.3,linewidth=0)
    for i in (start5_list):
        pl.axvspan(i, i + width, facecolor='gray', alpha=0.3,linewidth=0)
    for i in (start6_list):
        pl.axvspan(i, i + width, facecolor='orange', alpha=0.3,linewidth=0)
    for i in (start7_list):
        pl.axvspan(i, i + width, facecolor='cyan', alpha=0.3,linewidth=0)
    for i in (start8_list):
        pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)
    for i in (start9_list):
        pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)
    pl.xlim([0,test_len])
    pl.ylim([-0.1,1.1])
#plt.xlabel("Time [ms]", fontsize=11)
#plt.ylabel("Activity of soma [a.u.]", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.savefig('dend3.pdf', fmt='pdf', dpi=350)
