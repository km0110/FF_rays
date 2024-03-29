#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import imageio
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

font = {"family": "normal", "size": 15}

plt.rc("font", **font)


# In[2]:


def create_starting_optic(r, R, k=-1, N=100):
    """
    Creates the starting optic for the raytrace
    r: array of radii
    R: radius of curvature
    k: conic constant
    N: number of points in the optic

    returns: array of r and z coordinates of the optic
    """

    r = np.linspace(
        0, r, N
    )  # solves the problem of not having enough points close to the origin
    z = r * r / (R + np.sqrt(R * R - (k + 1) * r * r))
    optic = np.array([r, z])
    return optic


def find_local_eq(h, optic, N=100):
    """
    Finds the local equation of the optic around the point where the ray intersects

    h: height of the ray
    optic: array of r and z coordinates of the optic
    N: number of points to interpolate

    returns: cubic spline of the local optic
    """

    # first find nearest point in the lens array to where the ray r intersects
    r = optic[0]
    z = optic[1]
    index = np.abs(r - h).argmin()

    # isolate a few points around the closest index (look into how many points we actually want)
    lower = index - 3 if index - 3 > 0 else 0  # set the boundary conditions...
    upper = index + 3

    local_r = np.array(r[lower:upper]) - h
    local_z = np.array(z[lower:upper])

    # Use cubic spline to interpolate the local points
    # need to switch the z and the r coordinates so that cubic spline won't give error
    cs = None
    try:
        cs = CubicSpline(local_r, local_z)
    except:
        print(local_opt)
        print(lower)
        print(upper)
    # zs=np.linspace(local_z[0],local_z[-1],N)
    return cs


def find_reflect_slope(norm):
    """
    Finds the slope of the reflected ray

    norm: normal of the optic at the point of intersection

    returns: slope of the reflected ray
    """

    theta = np.arctan(norm)
    slope = np.tan(2 * theta)
    return slope


def raytrace(optic, exp_f, Nr=7, linsp=True):
    """
    Raytraces the optic

    optic: array of r and z coordinates of the optic
    exp_f: focal point of the optic
    Nr: number of rays to trace
    linsp: whether to use linear or logarithmic spacing for the rays. Default is True.

    returns: array of the rays and the points where they intersect the focal plane
    """

    # create the starting rays
    r = optic[0]
    # make sure that the rays are bounded
    r_max = r[-1]

    rays = (
        np.linspace(0, r_max, Nr) if linsp else np.geomspace(1e-9, r_max, Nr)
    )  # confine the rays to the diameter of the optic
    # rays[rays==0]=1e-9 # if r=0 exists set to small value so we don't get infinity values
    raymatrix = []  # 3 points: before, at, after the optic
    after = []
    for h in rays:
        cs = find_local_eq(h, optic)
        z_optic = cs(0)
        norm = 1 / cs(0, 1)  # The normal is just the derivative
        slope = find_reflect_slope(norm)
        r_after = slope * (exp_f - z_optic) + h  # This is where the ray meets z=exp_f
        # ray_z=[z_optic,exp_f]
        # ray_r=[h,r_after]
        raymatrix.append([h, r_after, z_optic, exp_f])
        after.append(r_after)
        # np.concatenate(raymatrix)
    return np.array(raymatrix), np.array(after)


def plot_raytrace(
    optic, raymatrix, exp_f, title=None, lambda0=None, norm=False, zoom=False
):
    """
    Plots the raytrace and saves the plot

    optic: array of r and z coordinates of the optic
    raymatrix: array of the rays and the points where they intersect the focal plane
    exp_f: focal point of the optic
    title: title of the plot
    lambda0: wavelength of the light
    norm: whether to normalize the plot
    zoom: whether to zoom in on the plot at the focal point
    """

    # first plot the optic:
    plt.figure(figsize=(15, 10))
    opt_r = optic[0] if not norm else optic[0] / lambda0
    opt_z = optic[1] if not norm else optic[1] / lambda0
    plt.plot(opt_z, opt_r, "k", opt_z, -1 * opt_r, "k")
    exp_freq = exp_f if not norm else exp_f / lambda0
    plt.axvline(x=exp_freq, color="k", linestyle="--")
    # Then plot the rays:
    plt.gca().set_prop_cycle(
        plt.cycler("color", plt.cm.RdYlGn(np.linspace(0, 1, 2 * len(raymatrix))))
    )
    for ray in raymatrix:
        ray_r = ray[0:2] if not norm else ray[0:2] / lambda0
        ray_z = ray[2:] if not norm else ray[2:] / lambda0
        plt.plot(ray_z, ray_r, ray_z, -1 * ray_r, alpha=0.75)

    xl = "z (m)" if not norm else "z/lambda"
    yl = "r (m)" if not norm else "r/lambda"
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.xlim((-0.01, exp_f + 0.01))
    r_max = max(opt_r) + 0.001
    plt.ylim((-r_max, r_max))
    plt.savefig(title + ".png")
    plt.close()

    # For zoomed in figure
    if zoom:
        plt.figure(figsize=(15, 10))
        plt.xlim((exp_f - 0.005, exp_f))
        plt.ylim((-0.005, 0.005))
        plt.gca().set_prop_cycle(
            plt.cycler("color", plt.cm.RdYlGn(np.linspace(0, 1, 2 * len(raymatrix))))
        )

        for ray in raymatrix:
            ray_r = ray[0:2] if not norm else ray[0:2] / lambda0
            ray_z = ray[2:] if not norm else ray[2:] / lambda0
            plt.plot(ray_z, ray_r, ray_z, -1 * ray_r, alpha=0.75)
        plt.xlabel("iterations")
        plt.ylabel("r(m)")
        plt.title(title + " zoom")
        plt.savefig(title + "_zoom.png")
    plt.close()


# In[7]:


def rms(rays_after):
    """
    Finds the root mean square of the rays after they intersect the optic at the focal plane

    rays_after: array of the rays and the points where they intersect the focal plane

    returns: root mean square of the rays
    """

    n = len(rays_after)
    return np.sqrt(np.sum(rays_after**2) / n)


def grad(i, epsilon, optic, exp_f, Nr):
    """
    Finds the gradient of the cost function with respect to the i-th point in the optic

    i: index of the point in the optic
    epsilon: small number to take the derivative
    optic: array of r and z coordinates of the optic
    exp_f: focal point of the optic
    Nr: number of rays to trace

    returns: gradient of the cost function with respect to the i-th point in the optic
    """

    o_r = optic[0]
    o_z = optic[1]
    j = i + 1
    # o_z[j]+=signs[i]*epsilon
    o_z[j] += epsilon
    rm1, af1 = raytrace([o_r, o_z], exp_f, Nr)
    # o_z[j]-=2*signs[i]*epsilon
    o_z[j] -= 2 * epsilon
    rm2, af2 = raytrace([o_r, o_z], exp_f, Nr)
    c1 = rms(af1)
    c2 = rms(af2)
    return c1 - c2


def plot_cdz(title, cdz, o_r):
    """
    Plots the change in the z coordinates of the optic at each iteration

    title: title of the plot
    cdz: array of the change in the z coordinate of the optic with respect to the iterations
    o_r: array of the r coordinates of the optic
    """

    fig, ax = plt.subplots(figsize=(12, 10))
    zeros = np.zeros((len(cdz), 1))
    cdz = np.hstack((zeros, cdz))
    n_iter = len(cdz)
    if n_iter > 100:
        steps = len(cdz) // 100
        cdz = cdz[::steps]
    im = ax.imshow(np.array(cdz).T, origin="lower")
    cbar = ax.figure.colorbar(im, fraction=0.046, pad=0.04, label=r"$\Delta dz$")
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel("iterations")
    plt.ylabel("r(m)")
    No = len(cdz[0]) + 1
    y_positions = np.linspace(0, No, 6)
    step = No // 5
    y_labels = np.append(o_r[::step], o_r[-1])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["{:.3e}".format(y) for y in y_labels])

    if n_iter > 100:
        x_positions = np.linspace(0, 100, 6)
        x_labels = np.linspace(0, n_iter - 1, 6).astype(int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["{:d}".format(x) for x in x_labels])

    plt.tight_layout()
    plt.savefig(title + "/" + title + "_plot_cdz.png")
    plt.close()
    # plt.show()


def plot_diff(title, o_z, o_r, r, R):
    """
    Plots the difference in the z coordinate of the optic at each iteration in respect to the initial optic

    title: title of the plot
    o_z: array of the z coordinates of the optic
    o_r: array of the r coordinates of the optic
    r: radius of the optic
    R: radius of curvature of the optic
    """

    oz = np.copy(o_z) * -1
    fig, ax = plt.subplots(figsize=(12, 10))
    No = len(o_r)
    start_o = oz[0]
    end_o = create_starting_optic(r, R, k=-1, N=No)[1][1:]
    # total_change=end_o+start_o[1:]
    exp_o = np.tile(end_o, (len(o_z), 1))
    # print(exp_o[0])
    oz[:, 1:] += exp_o
    # oz[:,1:]/=total_change
    # print(oz[0])
    n_iter = len(oz)
    if n_iter > 100:
        steps = len(oz) // 100
        oz = oz[::steps]

    im = ax.imshow(np.array(oz).T, origin="lower")
    cbar = ax.figure.colorbar(
        im, fraction=0.046, pad=0.04, label="K=-1 - current optic"
    )
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel("iterations")
    plt.ylabel("r(m)")
    step = No // 5
    y_positions = np.linspace(0, No, 6)
    y_labels = np.append(o_r[::step], o_r[-1])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["{:.3e}".format(y) for y in y_labels])

    if n_iter > 100:
        x_positions = np.linspace(0, 100, 6)
        x_labels = np.linspace(0, n_iter - 1, 6).astype(int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["{:d}".format(x) for x in x_labels])

    plt.tight_layout()
    plt.savefig(title + "/" + title + "_plot_diff.png")
    plt.close()
    # plt.show()


def plot_rms(title, cost):
    """
    Plots the root mean square of the rays at each iteration

    title: title of the plot
    cost: array of the root mean square of the rays at each iteration
    """

    plt.figure(figsize=(15, 10))
    plt.plot(cost)
    plt.xlabel("iterations")
    plt.ylabel("rms")
    plt.tight_layout()
    plt.savefig(title + "/" + title + "_plot_rms.png")
    # plt.show()
    plt.close()


# In[12]:


def write_data(o_z, o_r, cost, cdz, grads, after, raym, title):
    """
    Writes the data to csv files

    o_z: array of the z coordinates of the optic
    o_r: array of the r coordinates of the optic
    cost: array of the root mean square of the rays at each iteration
    cdz: array of the change in the z coordinate of the optic with respect to the iterations
    grads: array of the gradients of the cost function with respect to the i-th point in the optic
    after: array of the rays and the points where they intersect the focal plane at each iteration
    raym: array of the rays at each iteration
    title: title of the run
    """

    np.savetxt(title + "/" + title + "_o_z.csv", o_z)
    np.savetxt(title + "/" + title + "_o_r.csv", o_r)
    np.savetxt(title + "/" + title + "_cost.csv", cost)
    np.savetxt(title + "/" + title + "_cdz.csv", cdz)
    np.savetxt(title + "/" + title + "_grads.csv", grads)
    np.savetxt(title + "/" + title + "_after.csv", after)
    np.savetxt(title + "/" + title + "_rm.csv", raym)


def gradient_descent(
    epsilon,
    dz,
    start_k,
    r,
    R,
    exp_f,
    learn_rate,
    n_iter=1000,
    tol=1e-6,
    No=100,
    Nr=1000,
    plt=False,
    zoom=False,
    title=None,
):
    """
    Performs the gradient descent algorithm

    epsilon: small number to take the derivative
    dz: small number to take the derivative
    start_k: starting conic constant
    r: radius of the optic
    R: radius of curvature of the optic
    exp_f: focal point of the optic
    learn_rate: learning rate of the algorithm
    n_iter: number of iterations
    tol: tolerance of the algorithm
    No: number of points in the optic
    Nr: number of rays to trace
    plt: whether to plot the raytrace at each iteration
    zoom: whether to zoom in on the plot at the focal point
    title: title of the run
    """

    # Set up the initial conditions using the given parameters
    start_time = time.time()
    start_o = create_starting_optic(r, R, k=start_k, N=No)
    o_r = start_o[0]
    o_z = np.array([start_o[1]])
    rm0, af0 = raytrace(start_o, exp_f, Nr)
    cost = [rms(af0)]
    n = 0

    # Create the title and the directory to save the plots
    title = "dz-%1.e_k-%.2f_eps-%.1e_lr-%.1e_No-%d_Nr-%d_N-%d" % (
        dz,
        start_k,
        epsilon,
        learn_rate,
        No,
        Nr,
        n_iter,
    )
    os.system("mkdir " + title)

    # Plot the initial raytrace
    if plt:
        plot_title = title + "/raytrace"
        os.system("mkdir " + plot_title)
        plot_raytrace(start_o, rm0, exp_f, plot_title + "/step_%d" % (n), zoom=True)
    diff = cost[0]
    dzs = np.ones(No - 1) * dz
    cdz = np.array(dzs)

    o = start_o[1]
    grads = []
    after = []
    raym = []

    # Perform the gradient descent algorithm
    while n < n_iter and abs(diff) > tol:
        n += 1
        # move each point in the optic randomly by dz except for the point at origin
        o[1:] += dzs
        o_z = np.vstack([o_z, o])
        rm, af = raytrace([o_r, o], exp_f, Nr)
        raym.append(rm)
        after.append(af)
        c = rms(af)
        cost.append(c)
        if plt:
            plot_title = title + "/raytrace"
            plot(start_o, rm, exp_f, plot_title + "/step_%d" % (n), zoom=True)
        gs = []
        for i in range(len(dzs)):
            g = grad(i, epsilon, [o_r, o], exp_f, Nr)
            step_size = learn_rate * g
            dzs[i] = dzs[i] - step_size
            gs.append(g)
        diff = c
        cdz = np.vstack([cdz, dzs])
        grads.append(gs)
        print("Step:%d  \t Cost: %E \t time: %s" % (n, c, time.time() - start_time))

    runtime = time.time() - start_time
    raym = np.array(raym)
    # print(raym.shape)
    raym = raym.reshape(raym.shape[0], -1)
    write_data(o_z, o_r, cost, cdz, grads, after, raym, title)
    plot_cdz(title, cdz, o_r)
    plot_diff(title, o_z, o_r, r, R)
    plot_rms(title, cost)
    print(title + " finished")
    print(time.strftime("%H:%M:%S", time.gmtime(runtime)))
