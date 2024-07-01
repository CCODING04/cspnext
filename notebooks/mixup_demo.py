# YOLOv5 MixUp

import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.cla()
    x = np.linspace(1, 100, 100, endpoint=True)
    x_hat = np.ones_like(x) * 32.
    ratio = np.random.beta(x_hat, x_hat)
    plt.plot(x, ratio, '-ro')
    plt.savefig('out.png')

if __name__ == "__main__":
    main()