import numpy as np
import matplotlib.pyplot as plt

def PositionEncoding(seq_len, d_model, n=10000):
    p = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in range(int(d_model / 2)):
            denominator = np.power(n, 2 * i / d_model)
            p[k, 2*i] = np.sin(k/denominator)
            p[k, 2*i+1] = np.cos(k/denominator)
    return p

def plot_pe(pe):
    plt.figure(figsize=(10, 6))
    for idnex in range(pe.shape[0]):
        position = np.arange(0, pe[idnex].shape[0], 1)
        plt.plot(
            position,
            pe[idnex],
            linestyle='-',
            linewidth=1,
        )
    plt.xlim(min(position), max(position))
    plt.ylim(0, max(pe[idnex])*1.1)
    plt.xlabel('position', fontsize=12)
    plt.ylabel('pe', fontsize=12)
    plt.title('Position Embedding', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('./pe_curve.png')

if __name__ == "__main__":
    p = PositionEncoding(seq_len=128, d_model=4, n=10000).transpose(1, 0)
    print(p.shape)
    # print(p)
    plot_pe(p)
