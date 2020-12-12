from scipy import stats
from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

sns.set(style="darkgrid")

np.random.seed(42)


class normalgamma:
    def __init__(self, N, iterations, actual_mu, lambda_, bN, lambdaN, centered_x=False, a0=0, b0=0, mu0=0, lambda0=0):
        self.a0 = a0
        self.b0 = b0
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.N = N
        self.lambda_ = lambda_
        self.actual_mu = actual_mu
        self.iterations = iterations
        self.bN = bN
        self.lambdaN = lambdaN
        self.get_x(centered_x)
        self.calculate_true_values()
        self.linespace()

    def get_x(self, centered=True):
        x = np.random.normal(self.actual_mu, np.sqrt(1/self.lambda_), self.N)
        if centered:
            x = x - x.mean()

        self.x_bar = x.mean()
        self.x = x

    def linespace(self):
        self.uList = np.linspace(self.muT - 1.5,
                                 self.muT + 1.5, 50)

        self.tauList = np.linspace(
            (self.aT / self.bT) - 1.5, (self.aT / self.bT) + 1.5, 50)

    def _aN(self):
        self.aN = self.a0 + (self.N + 1) / 2

    def _bN(self):
        mu2 = 1 / self.lambdaN + self.muN ** 2
        # self.bN = (self.b0 + self.lambda0 * (mu2 + self.mu0 ** 2 - 2 * self.muN * self.mu0) + 0.5 * np.sum(self.x ** 2 + mu2 - 2 * self.muN * self.x) )
        self.bN = self.b0 - (sum(self.x) + self.lambda0 * self.mu0) * self.muN \
            + 0.5 * (sum(self.x**2) + self.lambda0 * self.mu0**2
                     + (self.lambda0 + self.N)*mu2)
        return self.bN

    def _lambdaN(self):
        self.tau = self.aN / self.bN
        self.lambdaN = (self.lambda0 + self.N) * (self.tau)
        return self.lambdaN

    def _muN(self):
        self.muN = (self.lambda0 * self.mu0 + self.N *
                    self.x_bar) / (self.lambda0 + self.N)

    def iterativeInference(self):
        # self.lambdaN = 0.1
        # self.bN = 0.1
        self._aN()
        self._muN()
        self.old_bN = self.bN
        self.old_lambdaN = self.lambdaN
        thres = 1e-8
        self.n_iterations = 0

        for i in range(self.iterations):
            self._bN()
            self._lambdaN()
            if (abs(self.bN - self.old_bN < thres) and abs(self.lambdaN - self.old_lambdaN < thres)):
                break
            self.plotPosteriors()
            self.old_bN = self.bN
            self.old_lambdaN = self.lambdaN
            self.n_iterations += 1
        print("n_iterations ", self.n_iterations)
        print("a", self.aN)
        print("b", self.bN)
        print("mu", self.muN)
        print("lambda", self.lambdaN)

    def computeAproxPosterior(self, x, tau):
        # Pricesion is 1/std^2
        q_mu = stats.norm.pdf(x, self.muN, np.sqrt(1 / self.lambdaN))

        # scale parameter is supposed to be 1/lambda
        q_tau = stats.gamma.pdf(tau, self.aN, scale=(1 / self.bN))
        return q_mu * q_tau

    def calculate_true_values(self):
        self.muT = (self.lambda0*self.mu0 + self.N *
                    self.x_bar) / (self.lambda0 + self.N)
        # self.muT = self.x_bar
        self.lambdaT = self.lambda0 + self.N
        self.aT = self.a0 + (self.N / 2)
        self.bT = self.b0 + 0.5 * np.sum((self.x - self.x_bar) ** 2) + (
            self.lambda0 * self.N * (self.x_bar - self.mu0) ** 2) / (2 * (self.lambda0 + self.N))
        # self.bT = 0.5 * np.sum((self.x - self.x_bar) ** 2)
        print("muT ", self.muT)
        print("lambdaT ", self.lambdaT)
        print("aT ", self.aT)
        print("bT ", self.bT)

    def computeExactPosterior(self, mu, tau):
        return ((self.bT ** self.aT) * np.sqrt(self.lambdaT) / (gamma(self.aT) * np.sqrt(2 * np.pi))) \
            * tau**(self.aT - 0.5) * np.exp(-self.bT * tau) * np.exp(-0.5 * self.lambdaT * tau * ((mu - self.muT) ** 2))

    def plotExactPosterior(self):
        M, T = np.meshgrid(self.uList, self.tauList, indexing="ij")
        Z = np.zeros_like(M)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i][j] = self.computeExactPosterior(
                    self.uList[i], self.tauList[j])

        plt.contour(M, T, Z, colors='green')

    def plotAproxPosterior(self):
        M, T = np.meshgrid(self.uList, self.tauList, indexing="ij")
        Z = np.zeros_like(M)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i][j] = self.computeAproxPosterior(
                    self.uList[i], self.tauList[j])

        plt.contour(M, T, Z, colors='red')

    def plotPosteriors(self):
        fig, ax = plt.subplots()
        plt.xlabel("Mean")
        plt.ylabel("tau")
        self.plotAproxPosterior()
        self.plotExactPosterior()
        custom_lines = [Line2D([0], [0], color="red", lw=4),
                        Line2D([0], [0], color="green", lw=4)]
        ax.legend(custom_lines, ['Inferred', 'True'])
        plt.show()


def calculateVariationalInference():
    N = 20
    iterations = 20
    mean = 0
    lambda_ = 1
    bN = 0.1
    lambdaN = 0.1

    ngamma = normalgamma(N, iterations, mean, lambda_, bN, lambdaN)
    ngamma.iterativeInference()


if __name__ == "__main__":
    calculateVariationalInference()
