"""
Copyright 2022 Toshitake Asabuki

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

alpha = 1
theta= 1
beta = 5
@numba.njit(fastmath=True, nogil=True)
def g(x):
    
    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))
    return ans
    
    
theta_gate= 0.5
beta_gate= 5
def g_gate(x):

    ans = 1/(1+alpha*np.exp(beta_gate*(-x+theta_gate)))
    return ans
    
@numba.njit( parallel=True,fastmath=True, nogil=True)
def learning(w,V_star,PSP_star,eps,f):
    for i in numba.prange(len(w[:,0])):
        for l in numba.prange(len(w[0,:])):
            delta=(-(g(V_star[i]))+f[i])*PSP_star[l]
            
            w[i,l]+=eps[i]*delta*beta*(1-g(V_star[i]))

    return w
    
additive_rec = False #True if dendrite receives additive recurrent connection


chunk_list = [['a', 'b', 'c', 'd'],['d', 'c', 'b', 'a'],['b', 'd', 'a', 'c']]
n_chunk = len(chunk_list)

sym_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i', 'j', 'k', 'l']

N = 1200
width = 50
dt = 1
window = 300

mu = np.zeros(N)
mu_square = np.zeros(N)+1

mu_gat = np.zeros(N)
mu_square_gat = np.zeros(N)+1

gamma_mu=0.0003
gamma_mu_square=gamma_mu

trainings = 100
nsecs=width*window+width*len(chunk_list)*trainings*50
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)

tau =15
tau_syn = 5
tau_syn_rec = 5
gain=0.05
n_syn = 2000
n_in =  n_syn
PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)

PSP_rec = np.zeros(N)
PSP_unit_rec = np.zeros(N)
I_syn_rec = np.zeros(N)

g_L=1/tau
g_d_max=1.

w  = np.zeros((N,n_syn))#afferent connections
w_gat  = np.zeros((N,N))#gating recurrent connections
w_rec_add  = np.zeros((N,N))#additive recurrent


mask_w=np.zeros((N,n_syn))
mask_w_gat=np.zeros((N,N))
mask_w_add=np.zeros((N,N))

p_in=1
p_rec=1

#afferent connections
for i in range(N):
    for j in range(n_syn):
        if np.random.rand()<p_in:
            w[i,j]=np.random.randn()/np.sqrt(p_in*n_syn)*1
            mask_w[i,j]=1

#gating recurrent connections
for i in range(N):
    for j in range(N):
        if np.random.rand()<p_rec:
            w_gat[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
            mask_w_gat[i,j]=1

#additive recurrent connections
for i in range(N):
    for j in range(N):
        if np.random.rand()<p_rec:
            w_rec_add[i,j]=np.random.randn()/np.sqrt(p_rec*N)*1
            mask_w_add[i,j]=1
            
poisson_signal =10
poisson_noise = 0.
eps = 10**(-5)
eps2 = 10**(-4)*1

p_connect = 1
max_rate = poisson_signal
w_inh_max = 0.5/np.sqrt(N)

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

color_list=['violet','maroon','limegreen','orange','red','grey','green','blue','teal','olivedrab','violet','tomato','peru','y']

chunk_color=['darkorange','deepskyblue','limegreen']

pat_element=np.zeros((len(sym_list),n_in,width),dtype=bool)
r_sig=5
for k in range(len(sym_list)):
    for i in range(n_in):
        for j in range(width):
            if np.random.rand()<r_sig*dt*10**(-3):
                pat_element[k,i,j]=1

pat=np.zeros((len(chunk_list),n_in,width*4),dtype=bool)

for k in range(4):
    for m in range(len(chunk_list)):
        pat[m,:,k*width:(k+1)*width]=pat_element[sym_list.index(chunk_list[m][k]),:,:]

chunk = chunk_list[np.random.randint(n_chunk)]


plot_len=3000
test_len= width*(len(chunk_list[0])+8)*20*len(chunk_list)

m = 0

input_pref = np.zeros(n_in)

for i in range(n_in):
    input_pref[i] = np.random.randint(len(sym_list))

PSP_mat = np.zeros((N,n_syn))
symbol_pat=np.zeros((n_in,len(sym_list)),dtype=bool)
for i in range(n_in):
    symbol_pat[i,np.random.choice(np.arange(len(sym_list)), 1, replace = False)]=1
    
V_star=np.zeros(N)
PSP_star=np.zeros((N,n_syn))
random_start=0

print("")
print("***********")
print("Learning... ")
print("***********")
spike_mat=np.zeros((n_in,simtime_len),dtype=bool)
chunk_type=np.zeros(simtime_len)
start_time=[]
for i in tqdm(range(simtime_len), desc="[creating input]"):

   if i == random_start:
       type='random'
       
       random_width=np.random.randint(width,width*8)
       pat_start=i+random_width

   if i==pat_start:
       
       random_start=i+width*4
       start_time.append(i)
       
       type=np.random.randint(len(chunk_list))
       chunk_type[i]=type
   if type=='random':
       spike_mat[:,i] = np.zeros(n_in,dtype=bool)
       spike_mat[np.random.rand(n_in)<r_sig*dt*10**(-3),i]=1
   else:
       spike_mat[:,i] =pat[type,:,i-pat_start]


for i in tqdm(range(simtime_len), desc="[learning]"):

    id=spike_mat[:,i]

    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    
    V_dend = np.dot(w,PSP_unit)
    if additive_rec==True:
        V_dend+=np.dot(w_rec_add,PSP_unit_rec)

    mu +=gamma_mu*(V_dend-mu)
    mu_square += gamma_mu_square*((V_dend)**2-mu_square)
    V_dend_hat=(V_dend-mu) / np.sqrt((mu_square-mu**2))
    V_gat=np.dot(w_gat,PSP_unit_rec)
    mu_gat +=gamma_mu*(V_gat-mu_gat)
    mu_square_gat += gamma_mu_square*(V_gat**2-mu_square_gat)
    g_d_hat=g_gate((V_gat-mu_gat) / np.sqrt((mu_square_gat-mu_gat**2)))*g_d_max
    g_d=g_gate(V_gat)*g_d_max

    V_som = (1.0-dt/tau)*V_som +g_d_hat*(V_dend_hat-V_som)-np.dot(w_inh,PSP_unit_rec)/tau

    V_star = V_dend*g_d/(g_d+g_L)

    f=g(V_som)

    id_rec = np.random.rand(N)<f*gain

    if i>=window*width:
        w=learning(w,V_star,PSP_unit,eps*g_d/(g_L+g_d),f)
        w_gat=learning(w_gat,V_star,PSP_unit_rec, 1*eps2*V_dend*(g_L*g_d*(1-g_d/g_d_max))/(g_L+g_d)**2*1,f)
        if additive_rec==True:
            w_rec_add=learning(w_rec_add,V_star,1*PSP_unit_rec,eps*g_d/(g_L+g_d),f)
    
    I_syn_rec = (1.0 - dt / tau_syn_rec) * I_syn_rec
    I_syn_rec[id_rec]+=1/tau/tau_syn_rec
    PSP_rec = (1.0 - dt / tau) * PSP_rec + I_syn_rec
    PSP_unit_rec=PSP_rec*25


print("")
print("***********")
print("Testing... ")
print("***********")


chunk_count=np.zeros(n_chunk)
chunk_start = [[] for k in range(n_chunk)]

chunk_order=[]
for k in range(len(chunk_list)):
    chunk_order+=[k]*20
chunk_order_random=np.random.permutation(np.array(chunk_order))
start_time=[]
random_start=0
appear_count=0
spike_mat=np.zeros((n_in,test_len),dtype=bool)
for i in range(test_len):
    if i == random_start:
        type='random'
        
        random_width=np.random.randint(width,width*8)
        pat_start=i+random_width
        
    if i==pat_start:
        start_time.append(i)
        random_start=i+width*4
        type=chunk_order_random[appear_count]
        
        appear_count+=1
        if type!='random':
            chunk_count[type]+=1
            chunk_start[type].append(i)

    if type=='random':
        spike_mat[:,i] = np.zeros(n_in,dtype=bool)
        spike_mat[np.random.rand(n_in)<r_sig*dt*10**(-3),i]=1
    else:
        spike_mat[:,i] =pat[type,:,i-pat_start]

    if appear_count>=20*len(chunk_list):
        end_point=i
        break

PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)
I_syn_rec=np.zeros(N)
PSP_rec=np.zeros(N)


V_dend_list =np.zeros((N,test_len))
V_gat_list =np.zeros((N,test_len))
V_som_list=np.zeros((N,test_len))
V_dend = np.zeros(N)

V_som = np.zeros(N)
chunk_id = np.random.randint(n_chunk)
chunk = chunk_list[chunk_id]

m = 0

f_list = np.zeros((N,test_len))
g_d_list = np.zeros((N,test_len))

id = np.zeros((test_len,n_in),dtype=bool)
id_rec = np.zeros((test_len,N),dtype=bool)

for i in range(test_len):

    id[i,:] = spike_mat[:,i]

    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id[i,:]]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    
    V_dend = np.dot(w,PSP_unit)
    if additive_rec==True:
        V_dend+=np.dot(w_rec_add,PSP_unit_rec)

    mu +=gamma_mu*(V_dend-mu)
    mu_square += gamma_mu_square*((V_dend)**2-mu_square)
    V_dend_hat=(V_dend-mu) / np.sqrt((mu_square-mu**2))
    V_gat=np.dot(w_gat,PSP_unit_rec)
    
    mu_gat +=gamma_mu*(V_gat-mu_gat)
    mu_square_gat += gamma_mu_square*(V_gat**2-mu_square_gat)
    
    V_gat_hat=(V_gat-mu_gat) / np.sqrt((mu_square_gat-mu_gat**2))
    g_d_hat=g_gate(V_gat_hat)*g_d_max
    
    V_som = (1.0-dt/tau)*V_som +g_d_hat*(V_dend_hat-V_som)-np.dot(w_inh,PSP_unit_rec)/tau
    f = g(V_som)
    id_rec[i,:] = np.random.rand(N)<f*gain
    f_list[:,i]=f
    g_d_list[:,i]=g_d_hat
    V_dend_list[:,i] = V_dend_hat
    V_gat_list[:,i] = V_gat_hat
    V_som_list[:,i]=V_som
    I_syn_rec = (1.0 - dt / tau_syn_rec) * I_syn_rec
    I_syn_rec[id_rec[i,:]]+=1/tau/tau_syn_rec
    PSP_rec = (1.0 - dt / tau) * PSP_rec + I_syn_rec
    PSP_unit_rec=PSP_rec*25

    if i==end_point:
        break

print("")
print("***********")
print("Plotting.. ")
print("***********")
avg_chunk = np.zeros((len(chunk_list),N,width*4))


count=np.zeros(len(chunk_list))
for k in range(len(chunk_list)):
    for i in np.array(chunk_start[k]):
        if i+width*4<test_len:
            avg_chunk[k,:,:]+=f_list[:,i:i+4*width]

            count[k]+=1

for k in range(len(chunk_list)):
    avg_chunk[k,:,:]/=count[k]

assembly_id_list=[]
for k in range(len(chunk_list)):
    assembly_id_list.append([])

rand_assembly_id_list=[]

for i in range(N):
    max_list=np.zeros(len(chunk_list))
    for k in range(len(chunk_list)):
        max_list[k]=max(avg_chunk[k,i,:])
    if max(max_list)>0.:
        
        assembly_id_list[np.argmax(max_list)].append(i)

    else:
        rand_assembly_id_list.append(i)
assembly_size=np.zeros(len(chunk_list))
for k in range(len(chunk_list)):
    assembly_size[k]=len(assembly_id_list[k])

t=[]
for k in range(len(chunk_list)):
    t.append(np.zeros(int(assembly_size[k])))

assembly_chunk=np.zeros((int(np.sum(assembly_size)),width*4*len(chunk_list)))

for l in range(len(chunk_list)):#chunk
    n0=0
    for k in range(len(chunk_list)):#assembly
        
        for i in np.array(chunk_start[l]):
            if i+width*4<test_len:
                assembly_chunk[int(n0):int(n0)+int(assembly_size[k]),width*4*l:width*4*(l+1)]+=f_list[np.array(assembly_id_list[k]) , i:i+width*4]/count[l]
        n0+=assembly_size[k]

assembly_chunk_norm=np.zeros((int(np.sum(assembly_size)),width*4*len(chunk_list)))
for k in range(len(chunk_list)):#assembly
    max_vec = np.zeros(int(assembly_size[k]))
    min_vec = np.zeros(int(assembly_size[k]))
    for l in range(len(chunk_list)):#chunk
        
        for i in range(int(assembly_size[k])):
            max_vec[i] = np.max(assembly_chunk[i+int(np.sum(assembly_size[0:k])),:])
            min_vec[i] = np.min(assembly_chunk[+int(np.sum(assembly_size[0:k])),:])

        for i in range(int(assembly_size[k])):
            assembly_chunk_norm[i+int(np.sum(assembly_size[0:k])),width*4*l:width*4*(l+1)] = (assembly_chunk[i+int(np.sum(assembly_size[0:k])),width*4*l:width*4*(l+1)]-min_vec[i])/(max_vec[i]-min_vec[i])

for k in range(len(chunk_list)):
    for j in range(int(assembly_size[k])):
        arg = np.angle(np.dot(assembly_chunk[j+int(np.sum(assembly_size[0:k])),width*4*k:width*4*(k+1)],np.exp(np.arange(width*4)/(width*4)*2*np.pi*1j))/sum(assembly_chunk[j+int(np.sum(assembly_size[0:k])),width*4*k:width*4*(k+1)]))
        if arg<0:
            arg += 2*np.pi
        t[k][j] = test_len/(2*np.pi)*arg

index = np.zeros(N)
n0=0
for k in range(len(chunk_list)):
    
    index[n0:n0+len(assembly_id_list[k])] = np.array(assembly_id_list[k])[np.argsort(t[k])]
    n0+=len(assembly_id_list[k])

count=0
len_assembly_id_sum=0
for k in range(len(chunk_list)):
    len_assembly_id_sum+=len(assembly_id_list[k])

for i in range(N):
    if  i not in index:
        index[len_assembly_id_sum+count]=i
        count+=1

id_rec_sorted=np.zeros((test_len,N),dtype=bool)
f_list_sorted=np.zeros((N,test_len))
for i in range(N):
    id_rec_sorted[:,i] = id_rec[:,int(index[i])]
    f_list_sorted[i,:] = f_list[int(index[i]),0:test_len]

tspk,nspk = pl.nonzero(id_rec_sorted[0:plot_len]==1)
fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(111)
plt.scatter(tspk,nspk,c='k',s=0.6,linewidth=0)
for k in range(len(chunk_list)):
    for i in np.array(chunk_start[k]):
        if i<plot_len:

            pl.axvspan(i, min(i + width*4,plot_len), facecolor=chunk_color[k], alpha=0.3,linewidth=0)

pl.ylim([0,int(N)])
pl.xlim([0,plot_len])

plt.ylabel("Neuron id", fontsize=11)
plt.xlabel("Time [ms]", fontsize=11)
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
fig.subplots_adjust(bottom=0.25, left=0.12)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('raster_output.pdf', fmt='pdf', dpi=350)

fig = plt.figure(figsize=(1.7*len(chunk_list), 1.7*len(chunk_list)))
ax = fig.add_subplot(111)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

n0=0
for k in range(len(chunk_list)**2):
    
    ax = fig.add_subplot(len(chunk_list), len(chunk_list), k+1)
    cax=plt.imshow(assembly_chunk_norm[np.argsort(t[int(k/len(chunk_list))])+n0,4*width*(k%len(chunk_list)):4*width*(k%len(chunk_list))+4*width], interpolation='nearest', aspect="auto",origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])
    for l in range(3):
        ax.axvline(x=width*(l+1), ymin=0, ymax=N, color='w', linewidth=0.5,ls='dashed')
    pl.xlim([0,width*4])
    if k%len(chunk_list)==0:
        plt.ylabel("Assembly%s"%int(k/len(chunk_list)+1),fontsize=10)
        plt.yticks([0,assembly_size[int(k/len(chunk_list))]-1],["%d"%(n0+1),"%d"%(n0+assembly_size[int(k/len(chunk_list))])],fontsize=10)
    if k%len(chunk_list)==len(chunk_list)-1:
        
        n0+=int(assembly_size[int(k/len(chunk_list))])
    if k>=len(chunk_list)*(len(chunk_list)-1):
        plt.xlabel("Time [ms]",fontsize=10)

fig.subplots_adjust(left=0.17,bottom=0.2,right=0.9)
plt.savefig('assembly_map.pdf', fmt='pdf',dpi=350)


