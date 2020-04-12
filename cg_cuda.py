import logging
import numpy as np
import cupy as cp
import time 

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

n = 5000
tol = 10**(-6)

A_cpu = np.random.uniform(size=(n,n)).astype(np.float32)

A_cpu = np.dot(A_cpu, A_cpu.T)
A_cpu /= np.mean(A_cpu)
A_cpu += 0.1 * np.eye(n)

true_x = np.random.uniform(size=n).astype(np.float32)
b_cpu = np.dot(A_cpu, true_x).astype(np.float32)

A_gpu = cp.asarray(A_cpu)
b_gpu = cp.asarray(b_cpu)

def solve(A, b):
    mod = cp.get_array_module(A)

    x = mod.zeros_like(b)

    resid = mod.dot(A, x) - b
    l2_sq = mod.inner(resid, resid)
    p = -resid
    iterations = 0

    start = time.time()

    while l2_sq > tol:
        iterations += 1

        A_p = mod.dot(A, p)
        p_A_p = mod.inner(p, A_p)
        alpha = l2_sq / p_A_p
        x += alpha * p
        new_resid = resid + alpha * A_p
        new_l2_sq = mod.inner(new_resid, new_resid)

        beta = new_l2_sq / l2_sq

        resid = new_resid
        l2_sq = new_l2_sq

        p = -resid + beta * p

    elapsed = time.time() - start
    logging.info(f'elapsed: {elapsed}')
    logging.info(f'resid. sq L2 norm: {l2_sq}')

print('CPU: ')
solve(A_cpu, b_cpu)
print('\n--------------------------\n')

print('GPU (not including data transfer): ')
solve(A_gpu, b_gpu)

