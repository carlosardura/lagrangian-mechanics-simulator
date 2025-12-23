import numpy as np 
import matplotlib.pyplot as plt

def eulode_phys(acc, tspan, pos0, vel0, h):
    
    t0, tf = tspan[0], tspan[1]
    if tf <= t0:
        raise ValueError("tspan must be strictly increasing")

    tp = np.arange(t0, tf + h, h) # same rounding problem as in "integrators.m"
    
    if tp[-1] < tf:
        tp = np.append(tp, tf)
    elif tp[-1] > tf:
        tp[-1] = tf
    n_out = len(tp)

    x = np.asarray(pos0, dtype=float).reshape(-1)
    v = np.asarray(vel0, dtype=float).reshape(-1)
    dim = x.size

    pos = np.zeros((n_out, dim))
    vel = np.zeros((n_out, dim))
    pos[0, :] = x
    vel[0, :] = v

    tt = t0

    for i in range(n_out - 1):
        t_end = tp[i + 1]
        while tt < t_end:
            hh = min(h, t_end - tt)
            v = v + hh * acc(x, v)
            x = x + hh * v
            tt += hh
        pos[i + 1, :] = x
        vel[i + 1, :] = v

    return tp, pos, vel