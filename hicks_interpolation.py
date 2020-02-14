import numpy as np
from scipy.special import iv

K_WIN = np.kaiser(8, 4.14)

def interpolate(nodes, position):
    radius = np.arange(-3, 5)
    refs = np.floor(position+radius).astype(np.int)
    val_idx = np.isin(refs, np.arange(len(nodes)))
    refs = refs[val_idx]
    x = np.abs(position - refs)
    delta = K_WIN[val_idx]*np.sinc(x)
    return (nodes[refs]*delta.reshape(-1, 1)).sum(axis=0)

