import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import theano.tensor as tt


def fit_spindle_density_prior():
    #data from purcell
    data = [[85, 177],
            [89, 148],
            [93, 115],
            [98, 71],
            [105, 42],
            [117, 20],
            [134, 17],
            [148, 27],
            [157, 39],
            [165, 53],
            [170, 68],
            [174, 84],
            [180, 102],
            [184, 123],
            [190, 143],
            [196, 156],
            [202, 165],
            [210, 173],
            [217, 176],
            [222, 177]]
    xscale = [0, 4]
    yscale = [0, 800]
    data_df = get_target_curve(data, xscale, yscale, scale=False)
    sample_data = np.random.choice(a=data_df['x'], p=data_df['y'], size=1000)
    with pm.Model() as model:
        a = pm.HalfNormal('a', 100*10)
        b = pm.HalfNormal('b', 100*10)
        pm.Beta('spindle_density', alpha=a, beta=b, observed=sample_data)
        trace = pm.sample(2000)
    summary_df = pm.summary(trace)
    a_est = summary_df.loc['a', 'mean']
    b_est = summary_df.loc['b', 'mean']

    n_samples = 10000
    with pm.Model() as model:
        pm.Beta('spindle_density_mean_params', alpha=a_est, beta=b_est)
        outcome = pm.sample(n_samples, njobs=1, nchains=1)
    # pm.traceplot(trace)
    # plt.show()
    samples = outcome['spindle_density_mean_params']
    sns.distplot(samples, kde=True)
    x = data_df['x']
    y = data_df['y']*len(samples)*(x[1]-x[0])
    sns.lineplot(x, y)
    plt.show()
    print(summary_df)

    sp_per_epoch = xscale[1]*outcome['spindle_density_mean_params']*25/60
    counts, bins, patches = plt.hist(sp_per_epoch, np.arange(0, 8)-0.5, density=True)
    sns.distplot(sp_per_epoch, kde=True, hist=False)
    plt.show()
    print(counts, bins)


def fit_spindle_duration():
    data = [
            [78, 163],
            [80, 30],
            [81, 15],
            [83, 6],
            [86, 8],
            [91, 26],
            [101, 51],
            [114, 85],
            [124, 105],
            [137, 126],
            [150, 139],
            [164, 150],
            [177, 156],
            [194, 160],
            [208, 163]
        ]
    xscale = [0.4, 2]
    yscale = [0, 4000]
    data_df = get_target_curve(data, xscale, yscale, scale=False)
    sample_data = np.random.choice(a=data_df['x'], p=data_df['y'], size=1000)
    with pm.Model() as model:
        a = pm.HalfNormal('a', 100*10)
        b = pm.HalfNormal('b', 100*10)
        pm.Gamma('spindle_duration', alpha=a, beta=b, observed=sample_data)
        trace = pm.sample(2000, njobs=1)
    summary_df = pm.summary(trace)
    a_est = summary_df.loc['a', 'mean']
    b_est = summary_df.loc['b', 'mean']

    n_samples = 10000
    with pm.Model() as model:
        pm.Gamma('spindle_density_mean_params', alpha=a_est, beta=b_est)
        outcome = pm.sample(n_samples, njobs=1, nchains=1)
    pm.traceplot(trace)
    plt.show()
    samples = outcome['spindle_density_mean_params']
    sns.distplot(samples, kde=True)
    x = data_df['x']
    y = data_df['y'] * len(samples) * (x[1] - x[0])
    sns.lineplot(x, y)
    plt.show()
    print(summary_df)
    return samples*(2-0.4)+0.4


def fit_spindle_refractory():
    data = [[88, 317],
                   [118, 99],
                   [125, 93],
                   [131, 97],
                   [137, 115],
                   [144, 143],
                   [151, 194],
                   [158, 223],
                   [175, 245],
                   [197, 265],
                   [239, 287],
                   [285, 297],
                   [355, 304],
                   [432, 307],
                   [454, 313]]
    xscale = [0, 30]
    yscale = [0, 0.08]
    data_df = get_target_curve(data, xscale, yscale, scale=False)
    sample_data = np.random.choice(a=data_df['x'], p=data_df['y'], size=1000)
    with pm.Model() as model:
        a = pm.HalfNormal('a', 100*10)
        b = pm.HalfNormal('b', 100*10)
        pm.Wald('spindle_duration', mu=a, lam=b, observed=sample_data)
        trace = pm.sample(2000, njobs=1)
    summary_df = pm.summary(trace)
    a_est = summary_df.loc['a', 'mean']
    b_est = summary_df.loc['b', 'mean']
    n_samples = 10000
    with pm.Model() as model:
        pm.Wald('spindle_density_mean_params', mu=a_est, lam=b_est)
        outcome = pm.sample(n_samples, njobs=1, nchains=1)
    # pm.traceplot(trace)
    # plt.show()
    samples = outcome['spindle_density_mean_params']
    sns.distplot(samples, kde=True, bins=100)
    x = data_df['x']
    y = data_df['y'] * len(samples) * (x[1] - x[0])
    sns.lineplot(x, y)
    plt.show()
    print(summary_df)
    return samples*30+0.5


def get_samples_for_refractory():
    samples = fit_spindle_refractory() + fit_spindle_duration()
    pd.DataFrame({'samples': samples}).to_pickle('../data/raw/refractory_prior_samples.pkl')


def fit_refractory_minus_duration():
    sample_data = pd.read_pickle('../data/raw/refractory_prior_samples.pkl')['samples'].values
    with pm.Model() as model:
        a = pm.HalfNormal('a', 100*10)
        b = pm.HalfNormal('b', 100*10)
        pm.Wald('prior', mu=a, lam=b, observed=sample_data)
        trace = pm.sample(2000, njobs=1)
    summary_df = pm.summary(trace)
    a_est = summary_df.loc['a', 'mean']
    b_est = summary_df.loc['b', 'mean']
    n_samples = 10000
    with pm.Model() as model:
        pm.Wald('prior_check', mu=a_est, lam=b_est)
        outcome = pm.sample(n_samples, njobs=1, nchains=1)

    samples = outcome['prior_check']
    sns.distplot(samples, kde=True)
    sns.distplot(sample_data, kde=True)
    plt.show()
    print(summary_df)


def get_target_curve(data, xscale=None, yscale=None, scale=True):
    df = pd.DataFrame(data, columns=['x','y'])
    df['y'] = df['y'].max() - df['y']
    ranges = df.agg(lambda x: x.max() - x.min())
    df = df - df.min()
    if scale:
        real_range = pd.Series([xscale[1] - xscale[0], yscale[1] - yscale[0]], index=['x','y'])
        real_offset = np.array([xscale[0],yscale[0]])
        x = np.linspace(*xscale, 99)
    else:
        real_range = np.array([1,1])
        real_offset = np.array([0,0])
        x = np.linspace(0, 1, 99)
    df = real_range*df/ranges + real_offset
    y = np.interp(x, df['x'], df['y'])
    df_interp = pd.DataFrame({'x':x,'y':y})
    df_interp['y'] /= df_interp['y'].sum()
    return df_interp