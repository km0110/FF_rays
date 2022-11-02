#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
font = {'family' : 'normal',
        'size'   : 15}

plt.rc('font', **font)


# In[2]:


def create_starting_optic(r,R,k=-1,N=100):
    r=np.linspace(0,r,N) #solves the problem of not having enough points close to the origin
    z=r*r/(R+np.sqrt(R*R-(k+1)*r*r))
    optic=np.array([r,z])
    return optic


# In[3]:


def find_local_eq(h,optic,N=100):
    # first find nearest point in the lens array to where the ray r intersects
    r=optic[0]
    z=optic[1]
    index=np.abs(r-h).argmin()
    
    # isolate a few points around the closest index (look into how many points we actually want)
    lower=index-3 if index-3>0 else 0 # set the boundary conditions...
    upper=index+3
    
    
    local_r=np.array(r[lower:upper])-h
    local_z=np.array(z[lower:upper])
    
    # Use cubic spline to interpolate the local points
    # need to switch the z and the r coordinates so that cubic spline won't give error
    cs=None
    try:
        cs=CubicSpline(local_r,local_z)
    except:
        print(local_opt)
        print(lower)
        print(upper)
    #zs=np.linspace(local_z[0],local_z[-1],N) 
    return cs


# In[4]:


def find_reflect_slope(norm):
    theta=np.arctan(norm)
    slope=np.tan(2*theta)
    return slope


# In[5]:


def raytrace(optic, exp_f, Nr=7, linsp=True):
    #create the starting rays
    r=optic[0]
    # make sure that the rays are bounded 
    r_max=r[-1]
    
    rays=np.linspace(0,r_max,Nr) if linsp else np.geomspace(1e-9,r_max,Nr) #confine the rays to the diameter of the optic
    #rays[rays==0]=1e-9 # if r=0 exists set to small value so we don't get infinity values
    raymatrix=[] # 3 points: before, at, after the optic
    after=[]
    for h in rays:
        cs=find_local_eq(h,optic)
        z_optic=cs(0)        
        norm=1/cs(0,1) #The normal is just the derivative 
        slope=find_reflect_slope(norm)
        r_after=slope*(exp_f-z_optic)+h # This is where the ray meets z=exp_f
        ray_z=[z_optic,exp_f]
        ray_r=[h,r_after]            
        raymatrix.append([ray_r,ray_z])
        after.append(r_after)
        #np.concatenate(raymatrix)
    return np.array(raymatrix),np.array(after)


# In[6]:


def plot(optic,raymatrix,exp_f,title=None, lambda0=None, norm=False,savefig=False):
    #first plot the optic:
    plt.figure(figsize=(15,10))
    opt_r=optic[0] if not norm else optic[0]/lambda0
    opt_z=optic[1] if not norm else optic[1]/lambda0
    plt.plot(opt_z,opt_r,'b',opt_z,-1*opt_r,'b')
    exp_freq=exp_f if not norm else exp_f/lambda0
    plt.axvline(x=exp_freq, color='k', linestyle='--')
    #Then plot the rays:
    for ray in raymatrix:
        ray_r=ray[0] if not norm else ray[0]/lambda0
        ray_z=ray[1] if not norm else ray[1]/lambda0
        plt.plot(ray_z,ray_r,'r',ray_z,-1*ray_r,'r')
        
    xl='z (m)' if not norm else 'z/lambda'
    yl='r (m)' if not norm else 'r/lambda'
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.xlim((-0.01,exp_f+0.02))
    r_max=max(opt_r)+0.001
    plt.ylim((-r_max,r_max))
    if savefig:
        plt.savefig(title+".png")
    #plt.show()
    plt.close()


# In[7]:


def rms(rays_after):
    n=len(rays_after)
    return np.sqrt(np.sum(rays_after**2)/n)


# In[8]:


def grad(i,epsilon,optic,exp_f,Nr):
    o_r=optic[0]
    o_z=optic[1]
    j=i+1
    # o_z[j]+=signs[i]*epsilon
    o_z[j]+=epsilon
    rm1,af1=raytrace([o_r,o_z],exp_f,Nr)
    # o_z[j]-=2*signs[i]*epsilon
    o_z[j]-=2*epsilon
    rm2,af2=raytrace([o_r,o_z],exp_f,Nr)
    c1=rms(af1)
    c2=rms(af2)
    return c1-c2


# In[9]:


def write_data(o_z,o_r,cost,cdz,title):
    os.system('mkdir '+title)
    np.savetxt(title+'/'+title+"_o_z.csv",o_z)
    np.savetxt(title+'/'+title+"_o_r.csv",o_r)
    np.savetxt(title+'/'+title+"_cost.csv",cost)
    np.savetxt(title+'/'+title+"_cdz.csv",cdz)


# In[26]:


def gradient_descent(epsilon,dz,start_k,thick,roc,exp_f,learn_rate,n_iter=1000,tol=1e-6,No=100,Nr=1000,plt=False,title=None):
    start_time=time.time()
    start_o=create_starting_optic(thick,roc,k=start_k,N=No)
    o_r=start_o[0]
    o_z=[start_o[1]]
    rm0,af0=raytrace(start_o,exp_f,Nr)
    cost=[rms(af0)]
    n=0
    if plt:
        plot(start_o,rm0,exp_f,title+"/step_%d"%(n),savefig=True)
    diff=cost[0]
    dzs=np.ones(No-1)*dz
    cdz=np.array(dzs)
    #print(dzs)
    #print('Step: %d\t Cost: %f'%(n,cost[0]))
    o=start_o[1]
    while(n<n_iter and abs(diff)>tol):
        #print(change_dzs)
        #start_time=time.time()
        n+=1
        #signs=np.random.choice([-1,1],No-1)
        #o[1:]+=signs*dzs #move each point in the optic randomly by dz except for the point at origin
        o[1:]+=dzs
        o_z.append(o)
        rm,af=raytrace([o_r,o],exp_f,Nr)
        c=rms(af)
        cost.append(c)
        if plt:
            plot([o_r,o],rm,exp_f,title+"/step_%d"%(n),savefig=True)
        for i in range(len(dzs)):
            step_size=learn_rate*grad(i,epsilon,[o_r,o],exp_f,Nr)
            dzs[i]=dzs[i]-step_size
        diff=c
        cdz=np.vstack([cdz,dzs])
        #print(dzs)
        #print('Step:%d  \t Cost: %E \t time: %s'%(n,c,time.time()-start_time))
    title='k-%.2f_eps-%.1e_lr-%.1e_No-%d_Nr-%d_N-%d_t-%.2e'%(start_k,epsilon,learn_rate,No,Nr,n_iter,time.time()-start_time)
    write_data(o_z,o_r,cost,cdz,title)
    #return o_z,o_r,cost,cdz

gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-3,n_iter=1000,Nr=20,No=20)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-3,n_iter=1000,Nr=50,No=50)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-3,n_iter=1000,Nr=250,No=250)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-3,n_iter=1000,Nr=500,No=500)

gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=50,No=100)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=100,No=100)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=250,No=100)
gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=500,No=100)

gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=50,No=100)
gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=100,No=100)
gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=250,No=100)
gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,1e-4,n_iter=1000,Nr=500,No=100)
