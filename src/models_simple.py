import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import theano.tensor as tt

data = pd.read_csv('markers.csv')
data = data.loc[data['epoch_i']<1,:] #Just do a single epoch for simplicity

# shapes and sizes
n_epochs = data['epoch_i'].max() + 1 #each epoch indexed by epoch_i
n_raters = data['rater_i'].max() + 1 #each rater indexed by rater_i
n_data = data.shape[0] #each spindle marker indexed by t

# static priors vars
trust_purcell = 0.1 #crank up to give more weight to purcell et al, 2017
purcell = np.array([0.3587, 0.6387, 0.0026, 0., 0., 0.])+(1-trust_purcell)
spindle_number_prior = purcell/purcell.sum()
max_spindles_per_epoch = len(spindle_number_prior)-1
spindle_duration_alpha = 0.975
spindle_duration_beta = 0.0899
duration_min = 0.4
duration_max = 2
spindle_refractory_mu = 8.81
spindle_refractory_lam = 14.91
spindle_refractory_prob_scale = 0.0339
expected_std_for_accuracy = 0.2

with pm.Model() as model:
    # --- Spindles come from True or Contamiate locations ---- #
    # Contaminates start and end can come from anywhere withing the 0-25 epoch
    contaminate_spindle_start = pm.Uniform.dist(lower=0., upper=25., shape=n_data)

    # True spindles in an epoch, must be ordered
    tss = pm.Uniform('true_spindle_starts', lower=0., upper=25., shape=max_spindles_per_epoch,
                     transform=pm.distributions.transforms.Ordered(),
                     testval=np.array([1., 5., 10., 15., 20.]).T)  # Real spindles

    # The number of spindles per epoch:
    num_spindles_per_epoch = pm.Categorical('num_spindles_per_epoch',
                                            p=pm.Dirichlet('spindle_num_prior', a=spindle_number_prior),
                                            testval=1)

    # ----Tracking is a raters spindle marker is real or contaminate-----
    # if the number of spindles in an epoch (z) is greater than 0, then use conf to determine if a spindle is real or not
    #spindle_chance = data['conf']  # pm.math.switch(num_spindles_per_epoch[data['epoch_i']] > 0, data['conf'], 0)
    spindle_chance_prior = pm.Beta('spindle_chance_prior', alpha=2, beta=1)
    marker_is_from_real_spindle = pm.Bernoulli('marker_is_from_real_spindle', p=spindle_chance_prior, shape=n_data)
    marker_is_from_real_spindle_stacked = tt.stack([marker_is_from_real_spindle, 1 - marker_is_from_real_spindle],
                                                   axis=1)  # stack theta for use in mixture model

    # ----Mapping between rater's spindles and real spindles (w)----
    ## Handy matrix to compare z too
    compare = np.arange(0, max_spindles_per_epoch + 1)  # [0 1 2 3 4 5]*epochs

    # Acutual prior for "mapping_marker_to_true_spindle"
    # shape=[n_epochs, max_spindles_per_epoch],
    # e.g. mapping_marker_to_true_spindle_prior for a single epoch will be like [1 1 1 0 0 0 0],
    # and therefore the mapping_marker_to_true_spindle's can only be from [0-2]-1 = [-1, 0, 1], where -1=no mapping
    mapping_marker_to_true_spindle_prior = pm.math.where(compare - num_spindles_per_epoch <= 0, 1, 0)
    # no_spindles_prior = np.zeros((n_data, 6))
    # no_spindles_prior[:, 0] = 1
    # mapping_prior = tt.switch(marker_is_from_real_spindle.reshape((n_data, 1)), mapping_marker_to_true_spindle_prior, no_spindles_prior)

    # Mapping between rater's spindles and true spindles, when=-1, marker_is_from_real_spindle will be 0, so we wont use it.
    mapping_marker_to_true_spindle = pm.Categorical('mapping_marker_to_true_spindle',
                                                    p=mapping_marker_to_true_spindle_prior,
                                                    shape=n_data, testval=0) - 1

    # --- Raters ability to detect true spindles --- #
    # rater_expertise = pm.Bound(pm.Normal, lower=0.)('rater_expertise',
    #                                                 mu=expected_std_for_accuracy,
    #                                                 sd=0.3,
    #                                                 shape=n_raters)

    # --- Observed behaviour --- #
    # Spindle start when marker is real
    mapping = mapping_marker_to_true_spindle[data['t']]
    spindle_real = pm.Normal.dist(mu=tss[mapping],
                                  sd=1)#rater_expertise[data['rater_i']])
    contaminate_spindle_start.mean = 12.5  # hack, https://discourse.pymc.io/t/how-to-use-a-densitydist-in-a-mixture/1371/2
    spindle_real.mean = 12.5
    obs_start = pm.Mixture('marker_start', w=marker_is_from_real_spindle_stacked,
                           comp_dists=[spindle_real, contaminate_spindle_start], observed=data['s'])

with model:
    trace = pm.sample(tune=1000, init="adapt_diag", nuts_kwargs={'target_accept': 0.99}) # turn off jitter so we dont break ordering gss

pm.traceplot(trace)
plt.show()
print(pm.summary(trace))