import numpy as np



def hanning_win():
    N = 480
    # print(N)
    window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * (n) / (N - 1)) for n in range(N)])
