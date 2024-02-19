import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

font = {"size": 15}

plt.rc("font", **font)


def create_starting_optic(thick, R, k=-1, N=100):
    """
    thick: float - thickness of the optic
    R: float - radius of curvature of the optic
    k: float - conic constant of the optic
    N: int - number of points to create the optic

    returns: np.array - array of shape (2,N) where the first row is the z coordinate and the second row is the r coordinate
    """
    z = np.geomspace(
        1e-6, thick, N
    )  # solves the problem of not having enough points close to the origin
    r = np.sqrt(2 * R * z - (k + 1) * z**2)
    optic = np.array([z, r])
    return optic


def find_local_eq(r, optic, N=1000):
    """
    r: float - the r coordinate of the point where we want to find the local equation
    optic: np.array - array of shape (2,N) where the first row is the z coordinate and the second row is the r coordinate
    N: int - number of points to interpolate the local equation

    returns: CubicSpline - the local equation of the optic
    """

    # first find nearest point in the lens array to where the ray r intersects
    z = optic[0]
    opt = optic[1]
    index = np.abs(opt - r).argmin()

    # isolate a few points around the closest index (look into how many points we actually want)
    lower = index - 5
    upper = index + 5

    local_z = np.array(z[lower:upper])
    local_opt = np.array(opt[lower:upper]) - r

    # Use cubic spline to interpolate the local points
    cs = CubicSpline(local_z, local_opt)
    return cs


def find_norm(z, cs):
    """
    z: float - the z coordinate of the point where we want to find the normal
    cs: CubicSpline - the local equation of the optic

    returns: float - the normal to the surface at point z"""

    # find the normal to the surface
    tang = cs(z, 1)  # 1st derivative of the spline at point z
    norm = -1 / tang
    return norm


def find_reflect_slope(norm):
    """
    Find the slope of the reflected ray

    norm: float - the normal to the surface at the point where we want to find the slope

    returns: float - the slope of the reflected ray
    """
    theta = np.arctan(norm)
    slope = np.tan(2 * theta)
    return slope


def find_refract_slope(n1, n2, slope):
    """
    Find the slope of the refracted ray

    n1: float - the index of refraction of the medium the ray is coming from
    n2: float - the index of refraction of the medium the ray is going to
    slope: float - the slope of the ray

    returns: float - the slope of the refracted ray
    """
    return n1 / n2 * slope


def raytrace(optic, Nr=7, refract=False):
    """
    Does the full raytrace from the optic to the optical axis

    optic: np.array - array of shape (2,N) where the first row is the z coordinate and the second row is the r coordinate
    Nr: int - number of rays to trace
    refract: bool - whether to trace refracted rays or reflected rays

    returns
    -------
    raymatrix: np.array - array of shape (Nr,3,2) where the first index is the initial z, the second index is where it intersects the optic and the third index is where it intersects the optical axis
    after: np.array - array of shape (Nr) where the first index is the ray number and the value is the r coordinate where the ray intersects the optical axis
    """
    # create the starting rays
    opt = optic[1]
    # make sure that the rays are bounded
    r_min = opt[5]
    r_max = opt[-10]

    # confine the rays to the diameter of the optic
    rays = np.linspace(r_min, r_max, Nr)
    # if r=0 exists set to small value so we don't get infinity values
    rays[rays == 0] = 1e-9
    raymatrix = []  # 3 points: before, at, after the optic
    after = []
    for r in rays:
        zs, cs = find_local_eq(r, optic)
        z_optic = cs.roots()
        if len(z_optic) > 1:
            print(
                "Warning: multiple intersections with lens found"
            )  # many roots are usually found near r=0

        norm = find_norm(z_optic[0], cs)
        slope = find_refract_slope(norm, 1, 3) if refract else find_reflect_slope(norm)
        z_after = (
            -r / slope + z_optic[0]
        )  # This is where the ray should intersect the z axis
        z_bef = (
            -1 if refract else z_after * 1.5
        )  # change this so that z_bef all starts at the same z value
        z_ray = [z_bef, z_optic[0], z_after]
        r_ray = [r, r, 0]
        raymatrix.append([z_ray, r_ray])
        after.append(z_after)
        # np.concatenate(raymatrix)
    return np.array(raymatrix), np.array(after)


def plot_raytrace(optic, raymatrix, title=None, lambda0=None, norm=False):
    """
    Plots the optic and the rays

    optic: np.array - array of shape (2,N) where the first row is the z coordinate and the second row is the r coordinate
    raymatrix: np.array - array of shape (Nr,3,2) where the first index is the initial z, the second index is where it intersects the optic and the third index is where it intersects the optical axis
    title: str - title of the plot
    lambda0: float - wavelength of the rays
    norm: bool - whether to normalize the rays or not
    savefig: bool - whether to save the figure or not
    """
    # first plot the optic:
    z_opt = optic[0] if not norm else optic[0] / lambda0
    r_opt = optic[1] if not norm else optic[1] / lambda0
    plt.plot(z_opt, r_opt)
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title("Optic")

    # plot the rays
    for ray in raymatrix:
        z_ray = ray[:, 0] if not norm else ray[:, 0] / lambda0
        r_ray = ray[:, 1] if not norm else ray[:, 1] / lambda0
        if norm:
            r_ray = r_ray / r_ray[0]
        plt.plot(z_ray, r_ray)
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title(title)

    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{title}.png")


# _________Histogram_________


def calc_bw(lambda0, diam, dec=6):
    """
    Calculate the bin width of the histogram based on the angular resolution of the system

    lambda0: float - wavelength of the rays
    diam: float - diameter of the optic
    dec: int - number of decimal places to round to

    returns: float - the angular resolution of the system
    """
    ang_res = 1.22 * lambda0 / diam  # The bin width is set as the minimum resolution
    return np.around(ang_res, dec)


def plot_hist(rays_after, exp_f, bins, norm=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(rays_after, bins)
    xl = "z (m)" if not norm else "z/lambda"
    ax.set_xlabel(xl)
    ax.ticklabel_format(useOffset=False)
    ax.set_xticks(bins)
    title = "Expected f: %f m" % (exp_f)
    ax.set_title(title)
    plt.savefig(title + "_hist.png")


def create_hist(rays_after, exp_f, bin_width, plot=False, lambda0=None, norm=False):
    """
    Creates a histogram of where the rays end on the optical axis

    rays_after: np.array - array of shape (Nr) where the first index is the ray number and the value is the r coordinate where the ray intersects the optical axis
    exp_f: float - the expected focal length
    bin_width: float - the width of the bins
    plot: bool - whether to plot the histogram or not
    lambda0: float - wavelength of the rays
    norm: bool - whether to normalize the rays or not

    returns: np.array - array of shape (k) where the first index is the bin number and the value is the number of rays in that bin

    """

    n = len(rays_after)
    # make k odd so that we have even number of bin below and above the expected focal value
    k = int(np.ceil(np.log2(n))) + 1
    if k % 2 == 0:
        k += 1

    bw = bin_width if not norm else bin_width / lambda0
    exp_freq = exp_f if not norm else exp_f / lambda0
    bin_low = exp_freq - bw * (0.5 * k)
    bin_high = exp_freq + bw * (0.5 * k)

    bins = np.linspace(bin_low, bin_high, k + 1)

    rays = rays_after if not norm else rays_after / lambda0
    # hist=np.histogram(rays,bins)
    hist, _ = np.histogram(rays, bins)

    # plots the histogram
    if plot:
        plot_hist(rays, exp_f, bin_width, lambda0, norm)

    h = np.array(hist)
    return h


# _________Cost Function_________


def chi_square(hist):
    """
    Calculate the chi square of the histogram

    hist: np.array - array of shape (k) where the first index is the bin number and the value is the number of rays in that bin

    returns: float - the chi square of the histogram
    """
    nr = np.sum(hist)
    Oi = hist / nr
    Ei = np.zeros(len(hist))
    index = int(np.floor(len(hist) / 2))
    Ei[index] = 1  # nr
    chi_square = np.sum((Oi - Ei) ** 2)
    return chi_square


def cost_function(hist, alpha, Nr):
    """
    Calculate the cost function of the histogram

    hist: np.array - array of shape (k) where the first index is the bin number and the value is the number of rays in that bin
    alpha: float - the weight of the cost function that characterizes how many rays end up in the histogram
    Nr: int - number of rays

    returns: float - the cost function of the histogram
    """
    return chi_square(hist) + alpha * (Nr - np.sum(hist)) / Nr


def run_raytrace(k, thick, roc, exp_f, lambda0, diam, Nr):
    """
    Run the full raytrace and create histogram of the rays that end up on the optical axis

    k: float - conic constant of the optic
    thick: float - thickness of the optic
    roc: float - radius of curvature of the optic
    exp_f: float - the expected focal length
    lambda0: float - wavelength of the rays
    diam: float - diameter of the optic
    Nr: int - number of rays to trace

    returns: np.array - array of shape (k) that is the histogram of the rays"""

    o = create_starting_optic(thick, roc, k=k)
    _, af = raytrace(o, refract=False, Nr=Nr)
    bw = calc_bw(lambda0, diam)
    hist = create_hist(af, exp_f, bw * 10)
    h = np.array(hist)
    return h


# _________Gradient_________


def grad(k, Del_k, alpha, thick, roc, exp_f, Nr):
    """
    Calculate the gradient of the cost function

    k: float - conic constant of the optic
    Del_k: float - step size for the gradient
    alpha: float - the weight of the cost function that characterizes how many rays end up in the histogram
    thick: float - thickness of the optic
    roc: float - radius of curvature of the optic
    exp_f: float - the expected focal length
    Nr: int - number of rays to trace

    returns: float - the gradient of the cost function
    """
    # Ray trace to slightly above and below k
    h1 = run_raytrace(k - Del_k, thick, roc, exp_f, Nr)
    h2 = run_raytrace(k + Del_k, thick, roc, exp_f, Nr)

    # Calculate the cost functions
    c1 = cost_function(h1, alpha, Nr)
    c2 = cost_function(h2, alpha, Nr)

    # return the gradient using central difference
    return (c2 - c1) / (2 * Del_k)
