import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def gibbs_sampler(n, n_burn, step):
    p1 = 0.444  # P(C=T | R=T,S=T,W=T)
    p2 = 0.048  # P(C=T | R=F,S=T,W=T)
    p3 = 0.815  # P(R=T | C=T,S=T,W=T)
    p4 = 0.216  # P(R=T | C=F,S=T,W=T)
    samples = np.zeros((2, n))
    last = np.ones(2)  # initialise: C=T, R=T
    i = 0
    for t in range(n * step + n_burn):
        sample_c = np.random.randint(0, 2)
        r = np.random.uniform(0, 1)
        if sample_c:
            last[0] = p1 > r if last[1] else p2 > r
        else:
            last[1] = p3 > r if last[0] else p4 > r
        if t > n_burn and (t - n_burn) % step == 0:
            samples[:, i] = last
            i += 1
    return samples


def plot_frequencies(data, datatype, sample_nr):
    frequencies = []
    last_sum = 0
    for i in range(len(data)):
        curr_sum = last_sum + data[i]
        frequencies.append(curr_sum / (i + 1))
        last_sum = curr_sum
    plt.plot(range(1, len(frequencies) + 1), frequencies)
    plt.title(f"Frequency of {datatype} = T in sample {sample_nr}")
    plt.xlabel("Iteration")
    plt.ylabel("Frequency")
    plt.show()


def plot_autocorrelation(data, datatype, sample_nr):
    sm.graphics.tsa.plot_acf(data)
    plt.title(f"Autocorrelation of {datatype} in sample {sample_nr}")
    plt.show()


def plot_n_estimations(n):
    estimates = []
    for _ in range(n):
        sample = gibbs_sampler(100, 10000, 10)
        estimate = sum(sample[1]) / len(sample[1])
        estimates.append(estimate)

    plt.plot(range(len(estimates)), estimates)
    plt.title(f"Estimation of P(R = T | S = T, W = T) for {n} independent samples")
    plt.xlabel("Sample")
    plt.ylabel("P(R = T | S = T, W = T)")
    plt.show()
