import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import theano.tensor as tt


def sample_ppc(trace, model, obs_vars):
    vars_to_sample = [var
                        for var in model.basic_RVs
                            for name in obs_vars.keys()
                                if name == var.name]
    vars_to_sample += model.observed_RVs
    ppc_samples = pm.sample_ppc(trace, 2000, model, vars=vars_to_sample)

    return ppc_samples


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out

    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])


def test(data):
    with pm.Model() as model:
        gs_s = pm.Uniform('gs_s', 0, 25)
        m_e = pm.Normal('m_e', mu=gs_s, sd=1, observed=data['e'])
    return model


def no_conf(data):
    n_gs = data['gs'].max()+1
    n_r = data['r'].max()+1
    expected_std_for_accuracy = 0.2
    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0)
        gs_s = pm.Uniform('gs_s', 0, 25, shape=(n_gs,))
        gs_e = pm.Uniform('gs_e', 0, 25, shape=(n_gs,))
        r_E = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=(n_r,))
        m_s = pm.Normal('m_s', mu=gs_s[data['gs']], sd=r_E[data['r']], observed=data['s'])
        m_e = pm.Normal('m_e', mu=gs_e[data['gs']], sd=r_E[data['r']], observed=data['e'])

    return model


def mixtureOfUniform(data):
    """
    z_i = [0-6], represents number of true spindles per epoch i
    r_t = [0-6], number of marked spindles per epoch per rater, known
    gss_i = [0-25]*zi, represents start point for true spindles
    gsd_i = [0.5-2]*zi, durations of spindles
    rs_t = [0-25]*r_t, starts, as marked by each rater
    rd_t = [0.5-2]*r_t, durations, as marked by each rater
    w_t = [0-z_i], mapping between r_t and z_i
    :param data:
    :return:
    """
    max_spindles_per_epoch = 6
    n_r = data['rater_i'].max()+1
    expected_std_for_accuracy = 0.2
    duration_av = 1
    duration_sd = 0.5
    n_i = len(data['epoch_i'].unique()) #only repping one, so this should be one
    n_t = data.shape[0]

    max_spindle_per_rater = data['spindle_i'].max()

    #set up model
    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        p = pm.Dirichlet('p', a=np.ones(max_spindles_per_epoch),
                         testval=[2]*max_spindles_per_epoch) #proportion of rater spindles to each real spindle. Uniform.
        gss = pm.Uniform('gss', lower=0., upper=25., shape=max_spindles_per_epoch,
                         testval=[1, 2, 6, 9, 12, 24])  # Real spindles
        # switching = tt.switch(gss[1] - gss[0] < 0, -np.inf, 0)
        # for i in range(1, max_spindles_per_epoch-1):
        #     switching += tt.switch(gss[i+1]-gss[i] < 0, -np.inf, 0)
        # pm.Potential('order', switching)
        comp_dists = gss.distribution
        comp_dists.mode = 12.5 #hack, https://discourse.pymc.io/t/how-to-use-a-densitydist-in-a-mixture/1371/2
        s = pm.Mixture('m_s', w=p, comp_dists=comp_dists, testval=12.5)
        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_r)
        raters_spindle_location = pm.Normal('raters_spindle_location',
                                            mu=s,
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))
    return model


def clustering_single_epoch(data):
    """
    z_i = [0-6], represents number of true spindles per epoch i
    r_t = [0-6], number of marked spindles per epoch per rater, known
    gss_i = [0-25]*zi, represents start point for true spindles
    gsd_i = [0.5-2]*zi, durations of spindles
    rs_t = [0-25]*r_t, starts, as marked by each rater
    rd_t = [0.5-2]*r_t, durations, as marked by each rater
    w_t = [0-z_i], mapping between r_t and z_i
    :param data:
    :return:
    """
    max_spindles_per_epoch = 6
    n_r = data['rater_i'].max()+1
    expected_std_for_accuracy = 0.2
    duration_av = 1.0
    duration_sd = 0.5
    n_t = data.shape[0]

    #set up model
    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        p = pm.Dirichlet('p', a=np.ones(max_spindles_per_epoch),
                         testval=[2]*max_spindles_per_epoch) #proportion of rater spindles to each real spindle.
        gss = pm.Uniform('gss', lower=0., upper=25., shape=max_spindles_per_epoch,
                         testval=[1, 2, 6, 9, 12, 24])  # Real spindles
        gsd = pm.Normal('gsd', mu=duration_av, sd=duration_sd, shape=max_spindles_per_epoch,
                         testval=[1, 1, 1, 1, 1, 1])  # Real spindles

        # Enforce ordering of markers
        switching = tt.switch(gss[1] - gss[0] < 0, -np.inf, 0)
        for i in range(1, max_spindles_per_epoch-1):
            switching += tt.switch(gss[i+1]-gss[i] < 0, -np.inf, 0)

        pm.Potential('order', switching)

        catergory_w = pm.Categorical('w', p=p, shape=n_t)

        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_r)

        raters_spindle_location = pm.Normal('s',
                                            mu=gss[catergory_w],
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

        raters_spindle_durations = pm.Normal('d',
                                            mu=gss[catergory_w]+gsd[catergory_w],
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['e'])

    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))
    return model


def clustering_multi_epoch_get_z(data):

    data = data.loc[data['epoch_i']<5,:]

    max_spindles_per_epoch = 6

    expected_std_for_accuracy = 0.2
    duration_av = 1.0
    duration_sd = 0.5
    n_epochs = data['epoch_i'].max() + 1
    n_raters = data['rater_i'].max() + 1
    n_data = data.shape[0]

    # data.drop('Conf')

    spindle_number_prior = np.array([5, 5, 6, 4, 3, 2, 1])

    with pm.Model() as model:

        # --- Real Spindle Parameters --- #
        contaminate_spindle = pm.Uniform('contaminate_spindle_dist', lower=0., upper=25.)
        contaminate_spindle_dur = pm.Normal('contaminate_spindle_dur', mu=duration_av, sd=5)
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_spindles_per_epoch),
                         testval=np.tile(np.array([1, 2, 6, 9, 12, 24]).T, reps=(n_epochs, 1)))  # Real spindles
        gsd = pm.Normal('gsd', mu=duration_av, sd=duration_sd, shape=(n_epochs, max_spindles_per_epoch))  # Real spindles

        z = pm.Categorical('z',
                           p=pm.Dirichlet('spindle_num_prior', a=spindle_number_prior),
                           shape=(n_epochs, )) #z is the number of real spindles (can be zero).
        #z = tt.printing.Print('z')(z1)
        z_rs = tt.reshape(z, newshape=(n_epochs, 1)) #reshape so we can compare

        # --- Mapping between raters spindles, and real spindles --- #
        z_compare_tiled = np.tile(np.arange(1, max_spindles_per_epoch+1), reps=(n_epochs, 1)) # shape=[n_epochs, max_spindles_per_epoch]
        z_cat_prior_per_epoch = pm.math.where(z_compare_tiled - z_rs <= 0, 1, 0)  # shape=[n_epochs, max_spindles_per_epoch], z_cat for a single epoch will be like [1 1 1 0 0 0], and therefore the w's can only be from 0-2
        w = pm.Categorical('w', p=z_cat_prior_per_epoch[data['epoch_i'], :], shape=n_data)

        # --- Conf and likelihood of a spindle being "real" --- #
        # The chance of a raters spindle mapping to a real spindle is proportional to how many possible spindles there are & confidence
        # spindle_chance = pm.Deterministic('spindle_chance',
        #                          0.5+0.5*(z[data['epoch_i']]-data['marker_per_r_i'])
        #                          /(z[data['epoch_i']]+data['marker_per_r_i']))

        spindle_chance = pm.math.switch(z[data['epoch_i']], data['conf'], 0) # if z is zero, then spindle chance is zero, otherwise its the raters confidence
        #spindle_chance = tt.printing.Print('spindle_chance')(spindle_chance1)
        theta = pm.Bernoulli('theta', p=spindle_chance, shape=n_data) #theta will be zero when z is zero
        #theta = tt.printing.Print('theta')(theta1)

        # --- Raters ability to detect markers --- #
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_raters)

        # --- Behaviour --- #
        spindle_mu_start = pm.math.switch(theta, contaminate_spindle, gss[data['epoch_i'], w[data['t']]])
        spindle_mu_end = pm.math.switch(theta, spindle_mu_start + contaminate_spindle_dur,
                                        gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]])

        raters_spindle_location = pm.Normal('s',
                                            mu=spindle_mu_start,
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

        raters_spindle_durations = pm.Normal('e',
                                             mu=spindle_mu_end,
                                             sd=rater_expertise[data['rater_i']],
                                             observed=data['e'])

        # The w's that are not zero need to be unique
        # And w should contain theta

        # ---- Constraints ---- #
        # Enforce ordering of markers
        switching = tt.switch(gss[:, 1] - gss[:, 0] < 0, -np.inf, 0)
        for i in range(1, max_spindles_per_epoch-1):
            switching += tt.switch(gss[:, i+1]-gss[:, i] < 0, -np.inf, 0)
        pm.Potential('order', switching)

    # for RV in model.basic_RVs:
    #     print(RV.name, RV.logp(model.test_point))

    return model


def cluster_multi_epoch_get_w(data, z):
    data = data.loc[data['epoch_i'] < 5, :]

    max_spindles_per_epoch = 6

    expected_std_for_accuracy = 0.2
    duration_av = 1.0
    duration_sd = 0.5
    n_epochs = data['epoch_i'].max() + 1
    n_raters = data['rater_i'].max() + 1
    n_data = data.shape[0]

    with pm.Model() as model:
        # --- Real Spindle Parameters --- #
        contaminate_spindle = pm.Uniform('contaminate_spindle_dist', lower=0., upper=25.)
        contaminate_spindle_dur = pm.Normal('contaminate_spindle_dur', mu=duration_av, sd=5)
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_spindles_per_epoch),
                         testval=np.tile(np.array([1, 2, 6, 9, 12, 24]).T, reps=(n_epochs, 1)))  # Real spindles
        gsd = pm.Normal('gsd', mu=duration_av, sd=duration_sd,
                        shape=(n_epochs, max_spindles_per_epoch))  # Real spindles

        z_rs = tt.reshape(z, newshape=(n_epochs, 1))  # reshape so we can compare

        # --- Mapping between raters spindles, and real spindles --- #
        z_compare_tiled = np.tile(np.arange(1, max_spindles_per_epoch + 1),
                                  reps=(n_epochs, 1))  # shape=[n_epochs, max_spindles_per_epoch]
        z_cat_prior_per_epoch = pm.math.where(z_compare_tiled - z_rs <= 0, 1,
                                              0)  # shape=[n_epochs, max_spindles_per_epoch], z_cat for a single epoch will be like [1 1 1 0 0 0], and therefore the w's can only be from 0-2
        w = pm.Categorical('w', p=z_cat_prior_per_epoch[data['epoch_i'], :], shape=n_data)

        spindle_chance = pm.math.switch(z[data['epoch_i']], data['conf'],
                                        0)  # if z is zero, then spindle chance is zero, otherwise its the raters confidence
        # spindle_chance = tt.printing.Print('spindle_chance')(spindle_chance1)
        theta = pm.Bernoulli('theta', p=spindle_chance, shape=n_data)  # theta will be zero when z is zero
        # theta = tt.printing.Print('theta')(theta1)

        # --- Raters ability to detect markers --- #
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_raters)

        # --- Behaviour --- #
        spindle_mu_start = pm.math.switch(theta, contaminate_spindle, gss[data['epoch_i'], w[data['t']]])
        spindle_mu_end = pm.math.switch(theta, spindle_mu_start + contaminate_spindle_dur,
                                        gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]])

        raters_spindle_location = pm.Normal('s',
                                            mu=spindle_mu_start,
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

        raters_spindle_durations = pm.Normal('e',
                                             mu=spindle_mu_end,
                                             sd=rater_expertise[data['rater_i']],
                                             observed=data['e'])

        # ---- Constraints ---- #
        # Enforce ordering of markers
        switching = tt.switch(gss[:, 1] - gss[:, 0] < 0, -np.inf, 0)
        for i in range(1, max_spindles_per_epoch - 1):
            switching += tt.switch(gss[:, i + 1] - gss[:, i] < 0, -np.inf, 0)

    return model


def cluster_multi_epoch_get_gss(data, w, theta):
    data = data.loc[data['epoch_i'] < 5, :]

    max_spindles_per_epoch = 6

    expected_std_for_accuracy = 0.2
    duration_av = 1.0
    duration_sd = 0.5
    n_epochs = data['epoch_i'].max() + 1
    n_raters = data['rater_i'].max() + 1

    with pm.Model() as model:
        # --- Real Spindle Parameters --- #
        contaminate_spindle = pm.Uniform('contaminate_spindle_dist', lower=0., upper=25.)
        contaminate_spindle_dur = pm.Normal('contaminate_spindle_dur', mu=duration_av, sd=5)
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_spindles_per_epoch),
                         testval=np.tile(np.array([1, 2, 6, 9, 12, 24]).T, reps=(n_epochs, 1)))  # Real spindles
        gsd = pm.Normal('gsd', mu=duration_av, sd=duration_sd,
                        shape=(n_epochs, max_spindles_per_epoch))  # Real spindles

        # --- Raters ability to detect markers --- #
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_raters)

        # --- Behaviour --- #
        spindle_mu_start = pm.math.switch(theta, contaminate_spindle, gss[data['epoch_i'], w[data['t']]])
        spindle_mu_end = pm.math.switch(theta, spindle_mu_start + contaminate_spindle_dur,
                                        gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]])

        raters_spindle_location = pm.Normal('s',
                                            mu=spindle_mu_start,
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

        raters_spindle_durations = pm.Normal('e',
                                             mu=spindle_mu_end,
                                             sd=rater_expertise[data['rater_i']],
                                             observed=data['e'])

        # ---- Constraints ---- #
        # Enforce ordering of markers
        switching = tt.switch(gss[:, 1] - gss[:, 0] < 0, -np.inf, 0)
        for i in range(1, max_spindles_per_epoch - 1):
            switching += tt.switch(gss[:, i + 1] - gss[:, i] < 0, -np.inf, 0)

    return model


def clustering_multi_epoch_w_priors(data, fit_for='z', fit_data=None):

    data = data.loc[data['epoch_i']<3,:] #SLICE DATA BC OTHERWISE IT TAKES TOO LONG!!

    expected_std_for_accuracy = 0.2
    duration_min = 0.4
    duration_max = 2
    n_epochs = data['epoch_i'].max() + 1
    n_raters = data['rater_i'].max() + 1
    n_rater_epochs = data['epoch_rater_i'].max() + 1
    n_data = data.shape[0]

    # static priors vars
    purcell = np.array([0.3587, 0.6387, 0.0026, 0., 0., 0.])+0.02
    spindle_number_prior = purcell/purcell.sum()
    max_spindles_per_epoch = len(spindle_number_prior)-1
    spindle_duration_alpha = 0.975
    spindle_duration_beta = 0.0899
    spindle_refractory_mu = 8.81
    spindle_refractory_lam = 14.91
    spindle_refractory_prob_scale = 0.0339

    # Need to add priors
    # a) end of previous cannot be greater than start of next

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)

        # --- Real Spindle Parameters --- #
        contaminate_spindle = pm.Uniform('contaminate_spindle_dist', lower=0., upper=25.)
        contaminate_spindle_dur = pm.Gamma('contaminate_spindle_dur',
                                           alpha=spindle_duration_alpha,
                                           beta=spindle_duration_beta)*(duration_max-duration_min)+duration_min
        gss = pm.Uniform('gss', lower=0., upper=25., shape=(n_epochs, max_spindles_per_epoch),
                         testval=np.tile(np.array([1., 5., 10., 15., 20.]).T, reps=(n_epochs, 1)))  # Real spindles
        gsd_gamma = pm.Gamma('gsd_gamma', alpha=spindle_duration_alpha,
                              beta=spindle_duration_beta,
                              shape=(n_epochs, max_spindles_per_epoch),
                              testval=0.5) # Real spindles
        gsd = pm.Deterministic('gsd', gsd_gamma*(duration_max-duration_min)+duration_min)

        if fit_for in ['z']: #when we are finding z, we need these
            z = pm.Categorical('z',
                           p=pm.Dirichlet('spindle_num_prior', a=spindle_number_prior),
                           shape=(n_epochs, )) #z is the number of real spindles (can be zero).
        else: #fir for w or gss
            z = fit_data['z']

        z_rs = tt.reshape(z, newshape=(n_epochs, 1))  # reshape so we can compare

        # --- Mapping between raters spindles, and real spindles --- #
        z_compare_tiled = np.tile(np.arange(1, max_spindles_per_epoch + 1),
                                  reps=(n_epochs, 1))  # shape=[n_epochs, max_spindles_per_epoch]

        # shape=[n_epochs, max_spindles_per_epoch], z_cat for a single epoch will be like [1 1 1 0 0 0],
        # and therefore the w's can only be from 0-2
        z_cat_prior_per_epoch = pm.math.where(z_compare_tiled - z_rs <= 0, 1,
                                              0)

        if fit_for in ['w','z']: #when we are finding z or w
            w = pm.Categorical('w', p=z_cat_prior_per_epoch[data['epoch_i'], :], shape=n_data)
            # if z is zero, then spindle chance is zero, otherwise its the raters confidence
            spindle_chance = pm.math.switch(z[data['epoch_i']], data['conf'], 0)
            theta = pm.Bernoulli('theta', p=spindle_chance, shape=n_data)  # theta will be zero when z is zero

        else: #fit for gss
            w = fit_data['w']
            theta = fit_data['theta']

        # --- Raters ability to detect markers --- #
        rater_expertise = BoundedNormal('r_E', mu=expected_std_for_accuracy, sd=0.25, shape=n_raters)

        # --- Behaviour --- #
        spindle_mu_start = pm.math.switch(theta, contaminate_spindle, gss[data['epoch_i'], w[data['t']]])
        spindle_mu_end = pm.math.switch(theta, spindle_mu_start + contaminate_spindle_dur,
                                        gss[data['epoch_i'], w[data['t']]] + gsd[data['epoch_i'], w[data['t']]])

        raters_spindle_location = pm.Normal('s',
                                            mu=spindle_mu_start,
                                            sd=rater_expertise[data['rater_i']],
                                            observed=data['s'])

        raters_spindle_durations = pm.Normal('e',
                                             mu=spindle_mu_end,
                                             sd=rater_expertise[data['rater_i']],
                                             observed=data['e'])

        # ---- unique gss assignment constraint ---- #
        if fit_for in ['w','z']:
            unique_count_table = tt.zeros(shape=(n_rater_epochs, max_spindles_per_epoch))
            unique_count_table = tt.set_subtensor(unique_count_table[data['epoch_rater_i'], w], theta)
            unique_q = tt.switch(unique_count_table<=1, 0, -np.inf)
            pm.Potential('w_uniqueness', unique_q)

        # ---- Refractory and order constraints ---- #
        gse = gss+gsd  # end of each spindle
        diff_e1_and_s2 = gss[:, 1:]-gse[:, :-1]  # difference between end of last one and start of next
        refact_p = wald_refractory_diff_contraint(diff_e1_and_s2,
                                                  spindle_refractory_mu,
                                                  spindle_refractory_lam)/spindle_refractory_prob_scale #scale the prob to between 0 and 1
        #refact_p = tt.printing.Print('refact')(refact_pp)
        refact_logp = -pm.math.log(refact_p)
        pm.Potential('refractory', tt.switch(z_cat_prior_per_epoch[:,:-1], refact_logp, 0))  # using the switch bc we dont want unused gss to contribute #TODO ask micheal, is 0 resonable??

    return model


def is_unique(x):
    return tt.switch(tt.sum(tt.extra_ops.Unique(False, False, True)(x) == tt.get_vector_length(x),1,0))


def refractory_constraint(prev_e, next_s, refactory_mu, refractory_lam):
    x = next_s-prev_e
    #Wald formula #TODO check shape!!
    return (pm.math.sqrt((refractory_lam/2*np.pi))*x**-3/2)*pm.math.exp((-refractory_lam/(2*x))*((x-refactory_mu)/refactory_mu)**2)


def wald_refractory_diff_contraint(x, refactory_mu, refractory_lam):
    return (pm.math.sqrt((refractory_lam / 2 * np.pi)) * x ** -3 / 2) * pm.math.exp(
        (-refractory_lam / (2 * x)) * ((x - refactory_mu) / refactory_mu) ** 2)


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




