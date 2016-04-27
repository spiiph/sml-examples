#!/usr/bin/env python
# encoding: utf-8
# Adopted from GaussianMixtureLearning/GMLearning.m

from time import sleep

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.special import psi

import numpy as np
from numpy.matlib import repmat


figures = []

params = {
    'figure.facecolor': 'white',
    'figure.subplot.bottom': 0.0,
    'font.size': 12,
    'lines.markeredgewidth': 2.0,
    'lines.linewidth': 0.5,
}
plt.rcParams.update(params)

# Gaussians for mixture (pi, mu, sigma)
mixture_parameters = [
    (0.3, np.array([4.0, 4.5]), np.array([[1.2, 0.6], [0.6, 0.6]])),
    (0.5, np.array([8.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])),
    (0.2, np.array([9.0, 8.0]), np.array([[0.6, 0.5], [0.5, 1.5]])),
]
n_mixtures = len(mixture_parameters)

# Number of samples for sampling
n_samples = 1000

# Number of components and iterations for optimization
n_components = 10
n_iterations = 50

# Number of standard deviations for contour plot of EM/VB components
n_std = 2

# Epsilon for when to discard a component
epsilon = 0.05

# Covariance epsilon for optimization
eps_cov = 1.0e-6 * np.eye(2)

x,y = np.mgrid[0:12:200j, -2:12:200j]

def initialize_gaussian(x, y, pi, mean, cov):
    xy = np.column_stack([x.flat, y.flat])
    z = pi * multivariate_normal.pdf(xy, mean, cov)
    return z.reshape(x.shape)

def check_shapes(x, y, mixtures, p):
    print('X shape: {}'.format(x.shape))
    print('Y shape: {}'.format(y.shape))
    for i, m in zip(range(len(mixtures)), mixtures):
        print('Mixture {} shape: {}'.format(i, m.shape))
    print('p shape: {}'.format(p.shape))


def plot(x, y, p, s):
    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x, y, p,
                    linewidth=0,
                    rstride=5,
                    cstride=5,
                    cmap=cm.Purples,
                    antialiased=False)
    ax1.set_title('PDF surface plot')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(x, y, p, cmap=cm.Purples)
    ax2.set_title('PDF contour plot')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    #ax3 = fig.add_subplot(2, 2, 3)
    #sx, sy = s.T
    #ax3.plot(sx, sy, 'k.', markersize=1.0, antialiased=False)
    #ax3.set_title('Sample')
    #ax3.set_xlabel('x')
    #ax3.set_ylabel('y')

    fig.tight_layout(pad=1.5)
    plt.show(block=False)
    #figures.append((fig, ax1, ax2, ax3))
    figures.append((fig, ax1, ax2))


def get_cov_ellipse(mean, cov):
    eigen_values, eigen_vectors = np.linalg.eigh(cov)

    # Compute the angle from the eigenvector corresponding to the largest
    # eigenvalue
    idx = eigen_values.argmax()
    angle = np.degrees(np.arctan2(eigen_vectors[idx,1], eigen_vectors[idx,0]))

    # Compute the width and height
    height, width = 2 * n_std * np.sqrt(np.abs(eigen_values))
    return mean, width, height, angle

def plot_ellipses(axes, distributions, s, iteration, title):
    # Get the x and y components of the samples
    sx, sy = s.T

    axes.clear()
    axes.set_title('{}, iteration {}, components {}'. \
                   format(title, iteration, len(distributions)))
    axes.plot(sx, sy, 'k.', markersize=1.0, antialiased=False)
    for pi,mean,cov in distributions:
        pos, w, h, a = get_cov_ellipse(mean, cov)
        axes.add_artist(Ellipse(pos, w, h, a,
                                ec='red',
                                ls='solid',
                                fc='none'))

def initialize_optimization_plot(x, y, s):
    # Get the x and y components of the samples
    sx, sy = s.T

    plt.ion()
    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sx, sy, 'k.', markersize=1.0, antialiased=False)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sx, sy, 'k.', markersize=1.0, antialiased=False)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout(pad=1.5)
    plt.show(block=False)
    figures.append((fig, ax1, ax2))
    return (fig, ax1, ax2)

def initialize_components():
    return zip(np.ones((n_components, 1))/n_components,
               [5.0, 5.0] + np.random.randn(n_components, 2) * 5.0,
               np.array([np.identity(2)]*n_components))

class VBParams():
    def __init__(self, mean, cov):
        # Initial parameters
        self.alpha0 = 1e-10
        self.beta0 = 0.01
        self.v0 = 0.01
        self.m0 = np.zeros(2)
        self.W0 = 1.0e5*np.identity(2)

        # NOTE: In the Bayesian framework, all the unknowns π, μ, Λ are
        # random and we choose conjugate priors
        #
        #   π ~ Dir(π|α₀) ∝ ∏(k=1:K) π_k^{α₀-1}
        #           − 10.39 Dirichlet distribution
        #
        #   μ,Λ ~ p(μ, Λ) ≡ ∏(k=1:K) N(μ_k|m₀, (β₀Λ_k)⁻¹) W(Λ_k|W₀,v₀)
        #           − 10.40 Gaussian-Wishart (for unknown mean and precision)

        # Continually updated parameters
        # In Bishop's notation, x_i is a measurement distributed according to
        #   x_i ~ p(x|π, μ, Λ) = ∑(k=1:K) π_k N(x|μ_k, Λ_k)
        # pi are the π in the optimization
        # l are the Λ (precisions) in the optimizations
        # NOTE: Where are the means in the optimization? Probably same as for
        # the posterior
        self.pi = np.ones(n_components) / n_components
        self.l = np.ones(n_components)

        # Parameters for the Dirichlet prior
        self.alpha = self.alpha0 * np.ones(n_components)

        # Parameters for the Gaussian-Wishart prior
        # Gaussian
        self.m = mean
        self.beta = self.beta0 * np.ones(n_components)
        # Wishart
        self.v = self.v0 * np.ones(n_components)
        self.W = cov

    def prune(self, ix):
        self.pi = self.pi[ix]
        self.l = self.l[ix]
        self.alpha = self.alpha[ix]
        self.m = self.m[ix]
        self.beta = self.beta[ix]
        self.v = self.v[ix]
        self.W = self.W[ix]


def optimize(x, y, s):
    # Get the x and y components of the samples
    sx, sy = s.T

    # Initialize the optimization plot
    fig, ax_em, ax_vb = initialize_optimization_plot(x, y, s)

    # Initialize components for expectation maximation in the form (pi, mu,
    # sigma). The means mu are random with mean 5.0 and variance
    # 25.0, while the covariances are eye(2).
    em = initialize_components()
    vb = initialize_components()
    vb_params = VBParams(np.array([c[1] for c in vb]),
                         np.array([c[2] for c in vb]))

    # Optimization
    for i in range(n_iterations):
        plot_ellipses(ax_em, em, s, i, 'EM')
        plot_ellipses(ax_vb, vb, s, i, 'VB')
        fig.tight_layout(pad=1.5)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Pause at the beginning of the loop
        sleep(0.01)

        # Expectation maximation
        # E-step
        # Compute the likelihoods that the sample points in s belong to each
        # of the k distributions
        r_em = np.array([pi*multivariate_normal.pdf(s, mean, cov + eps_cov)
                         for pi,mean,cov in em])
        # Normalize
        denom_em = np.sum(r_em, axis=0)
        r_em /= denom_em

        # M-step
        # The N_k normalization factors
        nk_em = np.sum(r_em, axis=1)

        # Compute the new pi's
        pi = nk_em/n_samples

        # Compute the new means
        mean = (np.dot(s.T, r_em.T) / nk_em).T

        # Compute the new covariances
        # NOTE: I want to do this by matrices, but not sure how or if at all
        # possible
        cov = []
        for r_em_k,nk_em_k,mean_k in zip(r_em, nk_em, mean):
            # Σ = 1/N_k sum_1^N r_em_k (x_n - mean_k) (x_n - mean_k)^T
            # NOTE: What happens with the cross terms: 2*x_n*mean_k?
            # NOTE: Why is there a minus sign on mean_k * mean_k^T?
            # NOTE: Why isn't mean_k * mean_k^T divided by 1/N_k?
            cov_k = np.dot(s.T*repmat(r_em_k, 2, 1), s)
            cov_k /= nk_em_k
            # [μ₀, μ₁]^T · [μ₀, μ₁] ⇒ 2x2 matrix
            mmT = np.asmatrix(mean_k).T*np.asmatrix(mean_k)
            cov_k -= mmT
            # Symmetrize
            cov_k = 0.5*(cov_k + cov_k.T)
            cov.append(cov_k)
        cov = np.array(cov)

        # Prune and compose new EM components
        em = [(pi_k, mean_k, cov_k)
              for pi_k, mean_k, cov_k
              in zip(pi, mean, cov)
              if pi_k > epsilon and np.all(np.linalg.eigvals(cov_k) > 0)]


        # Variational Bayes
        # E-step
        # Compute the likelihoods that the sample points in s belong to each
        # of the k distributions
        r_vb = []
        for W_k,pi_k,l_k,beta_k,v_k,m_k  \
                in zip(vb_params.W, vb_params.pi, vb_params.l,
                       vb_params.beta, vb_params.v, vb_params.m):
            # The Cholesky decomposition of W
            W_k_sqrt = np.linalg.cholesky(W_k + eps_cov)

            # Squared distance between the sample points and the means m
            d2 = np.sum(
                np.dot(W_k_sqrt,
                       (s.T - repmat(m_k, n_samples, 1).T))**2,
                axis=0)
            # Likelihoods of the sample points belonging to component k
            r_vb_k = pi_k * np.sqrt(l_k) * \
                np.exp(-2/2/beta_k - 0.5*v_k*d2)
            r_vb.append(r_vb_k)
        # NOTE: Inefficient
        r_vb = np.array(r_vb)

        # Normalize
        denom_vb = np.sum(r_vb, axis=0)
        r_vb /= denom_vb

        # M-step
        # The N_k normalization factors
        nk_vb = np.sum(r_vb, axis=1)

        # Compute the new pi's
        pi = nk_vb/n_samples

        # Compute the new means
        mean = (np.dot(s.T, r_vb.T) / nk_vb).T

        # Compute the new covariances
        # NOTE: I want to do this by matrices, but not sure how or if at all
        # possible
        cov = []
        for r_vb_k,nk_vb_k,mean_k in zip(r_vb, nk_vb, mean):
            # Σ = 1/N_k sum_1^N r_vb_k (x_n - mean_k) (x_n - mean_k)^T
            # NOTE: What happens with the cross terms: 2*x_n*mean_k?
            # NOTE: Why is there a minus sign on mean_k * mean_k^T?
            # NOTE: Why isn't mean_k * mean_k^T divided by 1/N_k?
            cov_k = np.dot(s.T*repmat(r_vb_k, 2, 1), s)
            cov_k /= nk_vb_k
            # [μ₀, μ₁]^T · [μ₀, μ₁] ⇒ 2x2 matrix
            mmT = np.asmatrix(mean_k).T*np.asmatrix(mean_k)
            cov_k -= mmT
            # Symmetrize
            cov_k = 0.5*(cov_k + cov_k.T)
            cov.append(cov_k)
        # NOTE: Inefficient
        cov = np.array(cov)

        # Update the parameters for the distributions of π, μ, Λ
        nk_nonzero = nk_vb.nonzero()
        vb_params.beta = vb_params.beta0 + nk_vb
        vb_params.v = vb_params.v0 + nk_vb + 1
        vb_params.alpha = vb_params.alpha0 + nk_vb

        # Means for the Gaussian part of the G-W distribution
        # NOTE: FFS. Maybe I should always use 2D ndarrays to avoid these
        # problems.
        vb_params.m = ((vb_params.beta0 * vb_params.m0 + (nk_vb*mean.T).T).T /
                       vb_params.beta).T
        # The W parameter for the Wishart part of the G-W distribution
        W = []
        for nk_vb_k, mean_k, cov_k in zip(nk_vb, mean, cov):
            mmT = \
                np.asmatrix(mean_k - vb_params.m0).T * \
                np.asmatrix(mean_k - vb_params.m0)
            W_k = np.linalg.inv(
                np.linalg.inv(vb_params.W0) + \
                nk_vb_k*cov_k + \
                vb_params.beta0 * nk_vb_k * mmT / (vb_params.beta0 + nk_vb_k))
            # Symmetrize
            W_k = 0.5*(W_k + W_k.T)
            W.append(W_k)
        vb_params.W = np.array(W)

        # The precision parameter (Λ) for the E-step
        # NOTE: psi is the di-gamma function
        l = []
        for W_k, v_k in zip(vb_params.W, vb_params.v):
            l_k = np.exp(np.sum(psi((v_k + 1 - np.array([1, 2]))/2)) + \
                         2*np.log(2) + \
                         np.log(np.linalg.det(W_k)))
            l.append(l_k)
        vb_params.l = np.array(l)
        # The partition probability (π) for the E-step
        vb_params.pi = np.exp(psi(vb_params.alpha) - psi(sum(vb_params.alpha)))

        # Find the indices of the components to keep
        ix = np.argwhere(
            np.array([pi_k > epsilon and
                      np.all(np.linalg.eigvals(cov_k) > 0)
                      for pi_k, cov_k
                      in zip(pi, cov)])
        ).flat

        # Prune and compose new VB components
        vb = [(pi_k, mean_k, cov_k)
              for pi_k, mean_k, cov_k
              in zip(pi[ix], mean[ix], cov[ix])]

        # Prune the optimization parameters
        vb_params.prune(ix)

    print('Final importances for EM: {}'.format([round(c[0], 3) for c in em]))
    print('Final importances for VB: {}'.format([round(c[0], 3) for c in vb]))


# Construct the combined distribution over a 200x200 mesh grid
mixtures = [initialize_gaussian(x, y, *p) for p in mixture_parameters]
combined_mixture = sum(mixtures[1:], mixtures[0])
#check_shapes(x, y, mixtures, p)

# Decide which distribution to sample from by chosing based on the
# probability pi from each distribution in the mixture
prior_probabilities = [p[0] for p in mixture_parameters]
which_distribution = np.random.choice(n_mixtures,
                                      n_samples,
                                      p=prior_probabilities)

# Get the partitioning
sample_partitioning = [sum(which_distribution == i) for i in range(n_mixtures)]
print('Sample partitioning for probabilities {}: {}'. \
      format(prior_probabilities, sample_partitioning))

# Perform the actual sampling and construct the combined sample
samples = [multivariate_normal.rvs(p[1], p[2], s)
           for p, s
           in zip(mixture_parameters, sample_partitioning)]
combined_sample = np.concatenate(samples)

# Plot the distribution as a surface and a contour plot
plot(x, y, combined_mixture, combined_sample)

optimize(x, y, combined_sample)

raw_input('Press any key to quit...')
