from raytrace_k import *
import time
from datetime import datetime


def main():
    # initial parameters
    start_k = -0.5
    Del_k = 0.001
    alpha = 1
    thick = 0.064
    roc = 0.1125
    exp_f = 0.05625
    learn_rate = 0.05
    n_iter = 500  # max number of iterations
    tol = 1e-6
    Nr = 1000
    lambda0 = 1.054e-6  # m
    diam = 0.075  # m

    # Start initial conditions
    start_time = time.time()
    k = start_k
    ks = [start_k]
    c1 = run_raytrace(start_k, thick, roc, exp_f, lambda0, diam, Nr)
    cost = [cost_function(c1, alpha, Nr)]
    n = 0
    diff = 1e6  # arbitrary large number

    # Gradient descent
    while n < n_iter and abs(diff) > tol and abs(k) < 8:
        diff = -learn_rate * grad(k, Del_k, alpha, thick, roc, exp_f, Nr)
        print("%d \t k:%f \t diff:%f" % (n, k, diff))
        k += diff
        ks.append(k)
        cost.append(cost_function(run_raytrace(k, thick, roc, exp_f, Nr), alpha, Nr))
        n += 1
    end_time = time.time()
    print("Time taken: %f" % (end_time - start_time))
    print("Final k: %f" % k)
    print("Number of iterations: %d" % n)


if __name__ == "__main__":
    main()
