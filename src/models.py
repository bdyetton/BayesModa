import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import theano.tensor as tt
Print = tt.printing.Print


def contaminate_mixture(data, fit_for='z', fit_data=None): #stickbreaking problems
    steps = []
    # shapes and sizes
    n_epochs = data['epoch_i'].max() + 1  # each epoch indexed by epoch_i
    n_raters = data['rater_i'].max() + 1
    n_obs = data.shape[0]  # each spindle marker indexed by t

    # static priors vars
    trust_purcell = 0.1  # crank up to give more weight to purcell et al, 2017
    purcell = np.array([0.3587, 0.6387, 0.0026, 0., 0., 0.]) + (1 - trust_purcell)
    s_number_prior = purcell / purcell.sum()
    max_s = len(s_number_prior) - 1
    gss_spindle_testvals = [1., 5., 10., 15., 20.]
    with pm.Model() as model:

        # True s
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_s),
                         testval=np.tile(np.array(gss_spindle_testvals).T, reps=(n_epochs, 1),))  # Real spindles
        gss_per_obs = gss[data['epoch_i'], :]

        # The number of spindles per epoch:
        if fit_for == 'z':
            gss_prior = pm.Dirichlet('gss_prior', a=s_number_prior)
            if n_epochs > 1:
                z = pm.Categorical('z', p=gss_prior,
                                   shape=n_epochs)
            else:
                z = pm.Categorical('z', p=gss_prior)
        else:
            z = fit_data['z']
        z_rs = z.reshape((n_epochs, 1))

        if fit_for in ['w', 'z']:  # when we are finding z or w
            w_prior_possibilities = tt.tril(tt.ones((max_s + 1, max_s + 1)))
            w = pm.Categorical('w', p=w_prior_possibilities[z_rs[data['epoch_i'], 0], :], shape=n_obs)
        else:  # fit for gss
            w = fit_data['w']

        # --- Raters ability to detect markers --- #
        r_E = pm.Bound(pm.Normal, lower=0.)('r_E', mu=0.5, sd=0.5, shape=n_raters)
        r_E_per_obs = r_E[data['rater_i']]
        #r_E = pm.Bound(pm.Normal, lower=0.)('r_E', mu=0.5, sd=0.5)

        # --- Behaviour --- #
        contaminate_dist_s = pm.Uniform.dist(lower=0., upper=25., shape=n_obs)
        contaminate_dist_s.mean = 12.5
        possible_dists = [contaminate_dist_s]
        for i in range(0, 5):
            dist = pm.Normal.dist(mu=gss_per_obs[:, i], sd=r_E_per_obs)
            dist.mean = gss_spindle_testvals[i]
            possible_dists.append(dist)

        w_array = tt.extra_ops.to_one_hot(w, nb_class=max_s + 1)
        s = pm.Mixture('s', w=w_array,
                       comp_dists=possible_dists,
                       observed=data['s'])

        #STEP methods for vars:
        if fit_for == 'z':
            steps = [pm.CategoricalGibbsMetropolis([z, w]),
                     pm.NUTS([gss_prior, gss, r_E], target_accept=0.9)]
        if fit_for == 'w':
            steps = [pm.CategoricalGibbsMetropolis([w]),
                     pm.NUTS([gss, r_E], target_accept=0.9)]
        #else, everything NUTS

    return model, steps


def wald_refractory_diff_contraint(x, refactory_mu, refractory_lam):
    wald_val = (pm.math.sqrt((refractory_lam / 2 * np.pi)) * x ** -3 / 2) * pm.math.exp(
        (-refractory_lam / (2 * x)) * ((x - refactory_mu) / refactory_mu) ** 2)
    return tt.switch(wald_val>0, wald_val, 0)


def wald_refractory_diff_contraint_np(x, refactory_mu, refractory_lam):
    return (np.sqrt((refractory_lam / 2 * np.pi)) * x ** -3 / 2) * np.exp(
        (-refractory_lam / (2 * x)) * ((x - refactory_mu) / refactory_mu) ** 2)


def plot_my_fun():
    def func(z, r):
        return 0.5*(z-r)/(r+z)+0.5

    data = np.zeros(shape=(3, 54))*np.nan
    i = 0
    X = np.arange(7)
    Y = np.arange(1, 10)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    # for z in range(7):
    #     for r in range(1, 10):
    #         data[:,i] = np.array([z,r,func(z,r)])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Axes3D.plot_surface(ax, X,Y,Z)
    plt.show()




