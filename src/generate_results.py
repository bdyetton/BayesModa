import pandas as pd
import numpy as np
import pickle
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mode


def extract_mode_as_array(trace, var='z', astype='array'):

    def trace_mode(x):
        return pd.Series(mode(x).mode[0], name='mode')

    df = pm.summary(trace, stat_funcs=[trace_mode], varnames=[var])
    df = df.reset_index()

    def split_fun(x):
        if '__' in x:
            return [int(x) for x in x.split('__')[1].split('_')]
        else:
            return [0]

    df['var type'] = df['index'].apply(lambda x: x.split('__')[0])
    df = df.loc[df['var type'] == var, :]
    var_idxs = df['index'].apply(split_fun)
    indexs = np.stack(var_idxs)
    if astype == 'array':
        sizes = indexs.max(axis=0) + 1
        var_array = df['mode'].copy().values.reshape(sizes)
        return var_array
    else:
        df_out = pd.DataFrame(np.concatenate([indexs, np.expand_dims(df['mode'].values, -1)], axis=1))
        df_out.columns = list(df_out.columns[:-1]) + [var]
        return df_out


def extract_mean_as_array(trace, var='z', astype='array'):
    df = pm.summary(trace)
    df = df.reset_index()

    def split_fun(x):
        if '__' in x:
            return [int(x) for x in x.split('__')[1].split('_')]
        else:
            return [0]

    df['var type'] = df['index'].apply(lambda x: x.split('__')[0])
    df = df.loc[df['var type'] == var, :]
    var_idxs = df['index'].apply(split_fun)
    indexs = np.stack(var_idxs)
    if astype == 'array':
        sizes = indexs.max(axis=0)+1
        var_array = df['mean'].copy().values.reshape(sizes)
        return var_array
    else:
        df_out = pd.DataFrame(np.concatenate([indexs, np.expand_dims(df['mean'].values, -1)], axis=1))
        df_out.columns = list(df_out.columns[:-1])+[var]
        return df_out


def plot_expertise(model_r, annotator_file):
    df = pm.summary(trace)
    df['e'] = df['mean']
    df['gs'] = df['variable']


def compare_to_gs_data(all_gs_data):
    fp = 0
    fn = 0
    tp = 0
    for gs, gs_data in all_gs_data.groupby('cluster'):
        if gs_data.shape[0] <= 2:
            print('Found a cluster of more than one matlab and one model, added a tp, and moving on')
        found_type = gs_data['Type'].values
        if 'matlab' in found_type and 'model' in found_type:
            tp += 1
        elif 'matlab' not in found_type and 'model' in found_type:
            fp += 1
        elif 'matlab' in found_type and 'model' not in found_type:
            fn += 1
        elif 'matlab' not in found_type and 'model' not in found_type:
            raise ValueError

    print('Recall:', tp/(tp+fn), 'Precision:', tp/(tp + fp))


def compare_r_data(r_data, annotator_data):
    r_data = pd.merge(r_data, annotator_data, on='annotatorID')
    sns.scatterplot(x='E', y='E_sd', hue='Scorer Type', data=r_data)
    plt.show()
    pass
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # p1 = sns.scatterplot(x=matlab_gs['s'], y=model_gs['s'])
    # r, p = pearsonr(matlab_gs['s'], model_gs['s'])
    # p1.text(0.5, 0.5, "r="+str(r)+', p='+str(p))
    # plt.subplot(1, 2, 2)
    # p2 = sns.scatterplot(x=matlab_gs['e'], y=model_gs['e'])
    # p2.text(0.5, 0.5, "r=" + str(r) + ', p=' + str(p))
    # plt.show()