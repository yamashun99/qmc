import itertools
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def jmc_single(t, L, spins, J, ntot, measure_interval):
    beta = 1.0/t
    Mzs = np.zeros(L**2*ntot//measure_interval)
    energys = np.zeros(L**2*ntot//measure_interval)
    measure_count = 0
    count = 0
    for i in range(ntot):
        for x in range(L):
            for y in range(L):
                deltaE = 2.0*J*spins[x, y]*(spins[(x+1) % L, y] + spins[(x-1) % L, y] +
                                            spins[x, (y+1) % L] + spins[x, (y-1) % L])
                if np.random.rand() < np.exp(-beta*deltaE):
                    spins[x, y] = -spins[x, y]
                if count % measure_interval == 0:
                    Mzs[measure_count] = np.sum(spins)/L**2
                    energys[measure_count] = measure_energy(L, J, spins)
                    measure_count += 1
                count += 1
    return spins, Mzs, energys


@njit
def get_cluster_number(index, cluster):
    i = index
    while i != cluster[i]:
        i = cluster[i]
    return i


@njit
def connect_bond(x1, y1, x2, y2, L, M, p, spins, cluster):
    if spins[x1, y1]*spins[x2, y2] < 0:
        return cluster
    if np.random.rand() > p:
        return cluster
    i1 = x1 + L*y1
    i2 = x2 + L*y2
    c1 = get_cluster_number(i1, cluster)
    c2 = get_cluster_number(i2, cluster)
    if c1 < c2:
        cluster[c2] = c1
    else:
        cluster[c1] = c2
    return cluster


@njit
def cluster_flip(L, M, px, ptau, spins, flip):
    cluster = np.arange(L*M)
    for x in range(L):
        for tau in range(M):
            cluster = connect_bond(x, tau, (x+1) %
                                   L, tau, L, M, px, spins, cluster)
            cluster = connect_bond(x, tau, x, (tau+1) %
                                   M, L, M, ptau, spins, cluster)
    for x in range(L):
        for tau in range(M):
            i = x + L*tau
            c = get_cluster_number(i, cluster)
            spins[x, tau] = flip[c]
    return spins


# @njit
def jmc_sw(t, L, M, spins, J, Hx, ntot, measure_interval):
    beta = 1.0/t
    Mzs = np.zeros(ntot//measure_interval)
    energys = np.zeros(ntot//measure_interval)
    tspins = np.zeros([ntot//measure_interval, L, M])
    measure_count = 0
    count = 0
    deltatau = beta/M
    px = 1/2*deltatau
    ptau = 1.0 - deltatau/2.0*Hx
    for i in range(ntot):
        flip = np.random.choice([-1, 1], size=(L * M))
        spins = cluster_flip(L, M, px, ptau, spins, flip)
        if count % measure_interval == 0:
            Mzs[measure_count] = np.sum(spins)/(L*M)
            energys[measure_count] = measure_energy(L, M, J, spins)
            tspins[measure_count] = spins
            measure_count += 1
        count += 1
    return spins, Mzs, energys, tspins


@njit
def measure_energy(L, M, J, spins):
    energy = 0.0
    for x in range(L):
        for y in range(M):
            energy += -J*spins[x, y]*(spins[(x+1) % L, y] + spins[(x-1) % L, y] +
                                      spins[x, (y+1) % M] + spins[x, (y-1) % M])
    return energy


class Mc:
    def __init__(self, t, L, M, J, Hx, ntot, measure_interval):
        self.t = t
        self.L = L
        self.J = J
        self.M = M
        self.Hx = Hx
        self.ntot = ntot
        self.measure_interval = measure_interval
        self.spins = np.random.choice([-1, 1], size=(self.L, self.M))

    def mc_single(self,):
        self.spins, Mzs, energys = jmc_single(
            self.t, self.L, self.spins, self.J, self.ntot, self.measure_interval)
        return self.spins, Mzs, energys

    def mc_sw(self,):
        self.spins, Mzs, energys, tspins = jmc_sw(
            self.t, self.L, self.M, self.spins, self.J, self.Hx, self.ntot, self.measure_interval)
        return self.spins, Mzs, energys, tspins

    def get_magnetization(self,):
        M = 0.0
        for x, y in itertools.product(range(self.L), repeat=2):
            M += self.spins[x, y]
        M = abs(M/self.L**2)
        return M

    def plot_spins(self,):
        fig = plt.figure()
        ax = fig.add_subplot()
        X, Y = np.meshgrid(range(self.L), range(self.M))
        # im = ax.pcolormesh(X, Y, self.spins.T, vmin=-1, vmax=1)
        im = ax.pcolormesh(X, Y, self.spins.T, cmap='binary', vmin=-1, vmax=1)
        # ax.set_aspect('equal')
        fig.colorbar(im, ax=ax)
        return fig, ax
