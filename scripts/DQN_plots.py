import numpy as np
import matplotlib.pyplot as plt

def plot_smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    plt.plot((cumsum[N:] - cumsum[:-N]) / float(N))
