import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import theano.tensor as tt
Print = tt.printing.Print

def clustering_multi_epoch_w_priors(data, fit_for='z', fit_data=None):

    # Add priors on expertise!

    data = data.loc[data['epoch_i']<4,:] #SLICE DATA BC OTHERWISE IT TAKES TOO LONG!!

    expected_std_for_accuracy = 0.2
    duration_min = 0.4
    duration_max = 2
    n_epochs = data['epoch_i'].max() + 1
    n_raters = data['rater_i'].max() + 1
    n_rater_epochs = data['epoch_rater_i'].max() + 1
    n_data = data.shape[0]

    # static priors vars
    trust_purcell = 0.1
    purcell = np.array([0.3587, 0.6387, 0.0026, 0., 0., 0.])+(1-trust_purcell)
    spindle_number_prior = purcell/purcell.sum()
    print(spindle_number_prior)
    max_spindles_per_epoch = len(spindle_number_prior)-1
    spindle_duration_alpha = 0.975
    spindle_duration_beta = 0.0899
    spindle_refractory_mu = 8.81
    spindle_refractory_lam = 14.91
    spindle_refractory_prob_scale = 0.0339

    with pm.Model() as model:

        # --- Real Spindle Parameters --- #
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_spindles_per_epoch),
                         transform=pm.distributions.transforms.Ordered(),
                         testval=np.tile(np.array([1., 5., 10., 15., 20.]).T, reps=(n_epochs, 1),))  # Real spindles
        gsd_gamma = pm.Gamma('gsd_gamma', alpha=spindle_duration_alpha,
                             beta=spindle_duration_beta,
                             shape=(n_epochs, max_spindles_per_epoch),
                             testval=0.5)  # Real spindles
        gsd = pm.Deterministic('gsd', gsd_gamma*(duration_max-duration_min)+duration_min)

        if fit_for in ['z']:  # when we are finding z, we need these
            z = pm.Categorical('z',
                               p=pm.Dirichlet('spindle_num_prior', a=spindle_number_prior),
                               shape=(n_epochs, ),
                               testval=1)  # z is the number of real spindles (can be zero).
        else:  # fit for w or gss
            z = fit_data['z']

        z_rs = tt.reshape(z, newshape=(n_epochs, 1))  # reshape so we can compare

        # --- Mapping between raters spindles, and real spindles --- #
        z_compare_tiled = np.tile(np.arange(0, max_spindles_per_epoch + 1),  # when w=0 means no spindles
                                  reps=(n_epochs, 1))  # shape=[n_epochs, max_spindles_per_epoch]

        # z_compare_tiled = np.tile(np.arange(1, max_spindles_per_epoch + 1), # when w=0 mean map to first spindle
        #                           reps=(n_epochs, 1))  # shape=[n_epochs, max_spindles_per_epoch]

        # shape=[n_epochs, max_spindles_per_epoch], z_cat for a single epoch will be like [1 1 1 0 0 0],
        # and therefore the w's can only be from 0-2
        z_cat_prior_per_epoch = pm.math.where(z_compare_tiled - z_rs <= 0, 1, 0)

        if fit_for in ['w', 'z']:  # when we are finding z or w
            # if z is zero, then spindle chance is zero, otherwise its the raters confidence
            #spindle_chance = pm.math.switch(z[data['epoch_i']] > 0, data['conf'], 0)
            #spindle_chance = pm.math.switch(w > 0, data['conf'], 0)
            theta = pm.Bernoulli('theta', p=data['conf'], shape=n_data) # theta will be zero when z is zero
            #w_zero_prior = np.tile(np.array([1, 0, 0, 0, 0, 0]).T, reps=(n_data, 1))
            # w_prior = tt.switch(tt.tile(theta, (max_spindles_per_epoch+1, 1)).T,
            #                     z_cat_prior_per_epoch[data['epoch_i'], :],
            #                     w_zero_prior)
            w = pm.Categorical('w', p=z_cat_prior_per_epoch[data['epoch_i'], :], shape=n_data, testval=0)-1

        else: #fit for gss
            w = fit_data['w']
            theta = fit_data['theta']

        # --- Raters ability to detect markers --- #
        rater_expertise = pm.Bound(pm.Normal, lower=0.)('r_E',
                                                        mu=expected_std_for_accuracy,
                                                        sd=0.3,
                                                        shape=n_raters)

        # --- Behaviour --- #
        # contaminate_spindle_s = pm.Uniform.dist(lower=0., upper=25., shape=n_data)
        # contaminate_spindle_e = pm.Uniform.dist(lower=0., upper=25., shape=n_data)
        #
        #
        # s_real = pm.Normal.dist(mu=gss[data['epoch_i'], w[data['t']]],
        #                         sd=rater_expertise[data['rater_i']])
        # e_real = pm.Normal.dist(mu=gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]],
        #                         sd=rater_expertise[data['rater_i']])
        #
        # contaminate_spindle_s.mean = 12.5
        # s_real.mean = 12.5
        # contaminate_spindle_e.mean = 13.5
        # e_real.mean = 13.5
        # obs_s = pm.Mixture('mix_s', w=theta_stacked, comp_dists=[s_real, contaminate_spindle_s], observed=data['s'])
        # obs_e = pm.Mixture('mix_e', w=theta_stacked, comp_dists=[e_real, contaminate_spindle_e], observed=data['e'])

        s_real = pm.Normal('s', mu=gss[data['epoch_i'], w[data['t']]], sd=rater_expertise[data['rater_i']], testval=12.5)
        e_real = pm.Normal('e', mu=s_real + gsd[data['epoch_i'], w[data['t']]], sd=rater_expertise[data['rater_i']], testval=13.5)

        contaminate_spindle_s = pm.Uniform('s_contam', lower=0., upper=25., shape=n_data, transform=None)
        contaminate_spindle_e = pm.Uniform('e_contam', lower=0., upper=25., shape=n_data, transform=None)
        marker_s = pm.math.switch(theta, s_real.logp, contaminate_spindle_s.logp)
        marker_e = pm.math.switch(theta, e_real.logp, contaminate_spindle_e.logp)

        pm.DensityDist('obs_s', logp=marker_s, observed=data['s'])
        pm.DensityDist('obs_e', logp=marker_e, observed=data['e'])


        # spindle_mu_start = pm.math.switch(theta,
        #                                   gss[data['epoch_i'], w[data['t']]],
        #                                   contaminate_spindle_s[data['t']])
        # spindle_mu_end = pm.math.switch(theta,
        #                                 gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]],
        #                                 contaminate_spindle_s[data['t']] + contaminate_spindle_dur[data['t']])
        #
        #
        # raters_spindle_location = pm.Normal('s',
        #                                     mu=spindle_mu_start,
        #                                     sd=spindle_sample_error,
        #                                     observed=data['s'])
        #
        # raters_spindle_durations = pm.Normal('e',
        #                                      mu=spindle_mu_end,
        #                                      sd=spindle_sample_error,
        #                                      observed=data['e'])

        # ---- unique gss assignment constraint ---- #
        # if fit_for in ['w', 'z']:
        #     unique_count_table = tt.zeros(shape=(n_rater_epochs, max_spindles_per_epoch))
        #    TODO theano scan
        #     unique_count_table = tt.set_subtensor(unique_count_table[data['epoch_rater_i'], w], theta)
        #     unique_q = tt.switch(unique_count_table <= 1, 0, -np.inf)
        #     pm.Potential('w_uniqueness', unique_q)

        # ---- Order & Refractory constraints ---- #
        # gse = gss + gsd  # end of each spindle
        # diff_e1_and_s2 = gss[:, 1:] - gse[:, :-1]
        # diff_flattened = diff_e1_and_s2.flatten()  # difference between end of last one and start of next
        # sliced_used_diffs_flattened = z_cat_prior_per_epoch[:, 1:-1].flatten().nonzero()[0]
        # used_diffs = diff_flattened[sliced_used_diffs_flattened]
        # pm.Potential('ordering', tt.switch(used_diffs < 0, -np.inf, 0))
        # refact_p = wald_refractory_diff_contraint(used_diffs,
        #                                           spindle_refractory_mu,
        #                                           spindle_refractory_lam) / spindle_refractory_prob_scale  # scale the prob to between 0 and 1
        # 
        # pm.Potential('refractory', pm.math.log(refact_p))  # using the switch bc we dont want unused gss to contribute

    #pm.model_to_graphviz(model)
    # plt.show(block=True)
    print(model.check_test_point())
    return model


def single_epoch_simple(data, fit_for='z', fit_data=None):
    data = data.loc[data['epoch_i'] == 0, :]  # Just do a single epoch for simplicity

    # shapes and sizes
    n_epochs = data['epoch_i'].max() + 1  # each epoch indexed by epoch_i
    n_raters = data['rater_i'].max() + 1  # each rater indexed by rater_i
    n_data = data.shape[0]  # each spindle marker indexed by t

    # static priors vars
    trust_purcell = 0.1  # crank up to give more weight to purcell et al, 2017
    purcell = np.array([0.3587, 0.6387, 0.0026, 0., 0., 0.]) + (1 - trust_purcell)
    spindle_number_prior = purcell / purcell.sum()
    max_spindles_per_epoch = len(spindle_number_prior) - 1
    start_z = 3
    with pm.Model() as model:
        #contaminate_spindle_start = pm.Uniform.dist(lower=0., upper=25., shape=n_data)

        # True spindles in an epoch, must be ordered
        tss = pm.Uniform('gss', lower=0., upper=25., shape=max_spindles_per_epoch,
                         transform=pm.distributions.transforms.Ordered(),
                         testval=np.array([1., 5., 10., 15., 20.]).T)  # Real spindles

        # The number of spindles per epoch:
        if fit_for == 'z':
            num_spindles_per_epoch = pm.Categorical('z', p=spindle_number_prior, testval=start_z)#p=pm.Dirichlet('spindle_num_prior', a=spindle_number_prior))
        else:
            num_spindles_per_epoch = fit_data['z']

        if fit_for in ['z', 'w']:
            # ----Tracking is a raters spindle marker is real or contaminate-----
            # if the number of spindles in an epoch (z) is greater than 0, then use conf to determine if a spindle is real or not
            # spindle_chance = data['conf']  # pm.math.switch(num_spindles_per_epoch[data['epoch_i']] > 0, data['conf'], 0)
            # spindle_chance_prior = pm.Beta('spindle_chance_prior', alpha=2, beta=1)
            # marker_is_from_real_spindle = pm.Bernoulli('theta', p=spindle_chance_prior, shape=n_data)
            # marker_is_from_real_spindle_stacked = tt.stack([marker_is_from_real_spindle, 1 - marker_is_from_real_spindle],
            #                                                axis=1)  # stack theta for use in mixture model

            # ----Mapping between rater's spindles and real spindles (w)----

            # Mapping between rater's spindles and true spindles, when=-1, marker_is_from_real_spindle will be 0, so we wont use it.
            mapping_marker_to_true_spindle_p = pm.DiscreteUniform('w', lower=0, upper=(num_spindles_per_epoch-1), shape=n_data, testval=np.random.randint(0,start_z,size=(n_data,)))
            mapping_marker_to_true_spindle = tt.printing.Print('w_printed')(mapping_marker_to_true_spindle_p)
        else:
            mapping_marker_to_true_spindle = fit_data['w']

        # marker_is_from_real_spindle = pm.math.neq(mapping_marker_to_true_spindle, 0)
        # marker_is_from_real_spindle_stacked = tt.stack([marker_is_from_real_spindle, 1 - marker_is_from_real_spindle],
        #                                                axis=1)

        # --- Raters ability to detect true spindles --- #
        # rater_expertise = pm.Bound(pm.Normal, lower=0.)('rater_expertise',
        #                                                 mu=expected_std_for_accuracy,
        #                                                 sd=0.3,
        #                                                 shape=n_raters)

        # --- Observed behaviour --- #
        # Spindle start when marker is real
        # spindle_real = pm.Normal.dist(mu=tss[mapping_marker_to_true_spindle-1],
        #                               sd=0.5)#rater_expertise[data['rater_i']])
        # contaminate_spindle_start.mean = 12.5  # hack, https://discourse.pymc.io/t/how-to-use-a-densitydist-in-a-mixture/1371/2
        # spindle_real.mean = 12.5
        # obs_start = pm.Mixture('marker_start', w=marker_is_from_real_spindle_stacked,
        #                        comp_dists=[spindle_real, contaminate_spindle_start], observed=data['s'])

        sd = pm.Bound(pm.Normal, lower=0)('sd', mu=0.5, sd=3)
        mapped_mu = tt.printing.Print('mm')(tss[mapping_marker_to_true_spindle])
        spindle_real = pm.Normal('marker_start', mu=mapped_mu, sd=sd, observed=data['s'])

    return model

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




