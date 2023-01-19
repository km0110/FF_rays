#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import imageio
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


def plot_cdz(title,cdz,o_r):
    fig, ax = plt.subplots(figsize=(12,10))
    zeros=np.zeros((len(cdz),1))
    cdz=np.hstack((zeros,cdz))
    n_iter=len(cdz)
    if n_iter>100:
        steps=len(cdz)//100
        cdz=cdz[::steps]
    im=ax.imshow(np.array(cdz).T,origin='lower')
    cbar=ax.figure.colorbar(im,fraction=0.046, pad=0.04, label=r'$\Delta dz$')
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel('iterations')
    plt.ylabel('r(m)')
    No=len(cdz[0])+1
    y_positions=np.linspace(0,No,6)
    step=No//5
    y_labels=np.append(o_r[::step],o_r[-1])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['{:.3e}'.format(y) for y in y_labels])
    
    if n_iter>100:
        x_positions=np.linspace(0,100,6)
        x_labels=np.linspace(0,n_iter-1,6).astype(int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['{:d}'.format(x) for x in x_labels])

    plt.tight_layout()
    plt.savefig(title+'/'+title+'_plot_cdz.png')
    #plt.show()


# In[10]:


def plot_diff(title,o_z,o_r,r,R):
    oz=np.copy(o_z)*-1
    fig, ax = plt.subplots(figsize=(12,10))
    No=len(o_r)
    start_o=oz[0]
    end_o=create_starting_optic(r,R,k=-1,N=No)[1][1:]
    #total_change=end_o+start_o[1:]
    exp_o=np.tile(end_o,(len(o_z),1))
    #print(exp_o[0])
    oz[:,1:]+=exp_o
    #oz[:,1:]/=total_change
    #print(oz[0])
    n_iter=len(oz)
    if n_iter>100:
        steps=len(oz)//100
        oz=oz[::steps]

    im=ax.imshow(np.array(oz).T,origin='lower')
    cbar=ax.figure.colorbar(im,fraction=0.046, pad=0.04,label='K=-1 - current optic')
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel('iterations')
    plt.ylabel('r(m)')
    step=No//5
    y_positions=np.linspace(0,No,6)
    y_labels=np.append(o_r[::step],o_r[-1])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['{:.3e}'.format(y) for y in y_labels])
    
    if n_iter>100:
        x_positions=np.linspace(0,100,6)
        x_labels=np.linspace(0,n_iter-1,6).astype(int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['{:d}'.format(x) for x in x_labels])

    plt.tight_layout()
    plt.savefig(title+'/'+title+ "_plot_diff.png")
    #plt.show()


# In[11]:


def plot_rms(title,cost):
    plt.figure(figsize=(15,10))
    plt.plot(cost)
    plt.xlabel('iterations')
    plt.ylabel('rms')
    plt.tight_layout()
    plt.savefig(title+'/'+title+"_plot_rms.png")
    #plt.show()
    plt.close()


# In[12]:


def write_data(o_z,o_r,cost,cdz,grads,after,title):
    np.savetxt(title+'/'+title+"_o_z.csv",o_z)
    np.savetxt(title+'/'+title+"_o_r.csv",o_r)
    np.savetxt(title+'/'+title+"_cost.csv",cost)
    np.savetxt(title+'/'+title+"_cdz.csv",cdz)
    np.savetxt(title+'/'+title+"_grads.csv",grads)
    np.savetxt(title+'/'+title+"_after.csv",after)


# In[13]:


def create_gif(title,n_iter):
    writer = imageio.get_writer(title+'/'+title+'_raytrace.mp4', fps=10)
    for i in range(n_iter+1):
        f=title+'/raytrace/step_%d.png'%i
        im=imageio.imread(f)
        writer.append_data(im)
    writer.close()


# In[14]:


def gradient_descent(epsilon,dz,start_k,r,R,exp_f,learn_rate,n_iter=1000,tol=1e-6,No=100,Nr=1000,plt=False,title=None):
    start_time=time.time()
    start_o=create_starting_optic(r,R,k=start_k,N=No)
    o_r=start_o[0]
    o_z=np.array([start_o[1]])
    rm0,af0=raytrace(start_o,exp_f,Nr)
    cost=[rms(af0)]
    n=0
    title='dz-%1.e_k-%.2f_eps-%.1e_lr-%.1e_No-%d_Nr-%d_N-%d'%(dz,start_k,epsilon,learn_rate,No,Nr,n_iter)
    os.system('mkdir '+title)

    if plt:
        plot_title=title+"/raytrace"
        os.system('mkdir '+plot_title)
        plot(start_o,rm0,exp_f,plot_title+"/step_%d"%(n),savefig=True)
    diff=cost[0]
    dzs=np.ones(No-1)*dz
    cdz=np.array(dzs)
    #print(dzs)
    #print('Step: %d\t Cost: %f'%(n,cost[0]))
    o=start_o[1]
    grads=[]
    after=[]
    while(n<n_iter and abs(diff)>tol):
        #print(change_dzs)
        #start_time=time.time()
        n+=1
        #signs=np.random.choice([-1,1],No-1)
        #o[1:]+=signs*dzs #move each point in the optic randomly by dz except for the point at origin
        o[1:]+=dzs
        o_z=np.vstack([o_z,o])
        rm,af=raytrace([o_r,o],exp_f,Nr)
        after.append(af)
        c=rms(af)
        cost.append(c)
        if plt:
            plot_title=title+"/raytrace"
            plot(start_o,rm,exp_f,plot_title+"/step_%d"%(n),savefig=True)
        gs=[]
        for i in range(len(dzs)):
            g=grad(i,epsilon,[o_r,o],exp_f,Nr)
            step_size=learn_rate*g
            dzs[i]=dzs[i]-step_size
            gs.append(g)
        diff=c
        cdz=np.vstack([cdz,dzs])
        #print(dzs)
        grads.append(gs)
        print('Step:%d  \t Cost: %E \t time: %s'%(n,c,time.time()-start_time))
    
    runtime=time.time()-start_time
    write_data(o_z,o_r,cost,cdz,grads,after,title)
    plot_cdz(title,cdz,o_r)
    plot_diff(title,o_z,o_r,r,R)
    plot_rms(title,cost)
    create_gif(title,n_iter)
    print(title + " finished")
    print(time.strftime("%H:%M:%S", time.gmtime(runtime)))
    #return np.array(o_z),o_r,cost,cdz


# Default 

# In[ ]:


#gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,0.03,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,0.03,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,0.02,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,-1.5,0.0375,0.1125,0.05625,0.02,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,0,0.0375,0.1125,0.05625,0.03,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,-2,0.0375,0.1125,0.05625,0.03,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,0,0.0375,0.1125,0.05625,0.02,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


#gradient_descent(1e-7,5e-7,-2,0.0375,0.1125,0.05625,0.02,n_iter=5000,Nr=20,No=20,plt=True)


# In[ ]:


# learn_rate=np.linspace(1e-2,1e-1,10)
# num_opt=[10,20,50,100]


# In[ ]:


# for no in num_opt:
#     for lr in learn_rate:
#         nr=no
#         gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,lr,n_iter=2000,Nr=nr,No=no)
#         nr=no/2
#         gradient_descent(1e-7,5e-7,-0.5,0.0375,0.1125,0.05625,lr,n_iter=2000,Nr=nr,No=no)

