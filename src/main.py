import matplotlib
matplotlib.use('TkAgg')
from src import plot_epoch, parse_data, models, generate_results, fitting_priors
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
sys.path.extend(['/home/bdyetton/BayesModa', 'C:/Users/bdyet/GoogleDrive/MODA/BayesModa/src'])


def create_and_save_dataset():
    """Load dataset and save for easy model fitting"""
    file_to_load = "../data/raw/spindle_markers_by_annotator_MODA.txt"
    markers, annot_map, epoch_i_map = parse_data.load_and_parse_for_modeling(file_to_load)
    pickle.dump((annot_map, epoch_i_map), open('../data/preprocessed/maps.pkl', 'wb'))
    markers.to_csv('../data/preprocessed/markers.csv', index=False)


def run_model(model_name):
    """fit model by name. Not used currently."""
    print('Fitting', model_name)
    data = pd.read_csv('../data/preprocessed/markers.csv')
    model = eval('models.'+model_name+'(data)')
    with model:
        trace = pm.sample(init="adapt_diag", nuts_kwargs={'target_accept':0.99})  # Jitter will likely break order constraints during training, so dont use
        pm.traceplot(trace)
    pickle.dump((model, trace), open('../data/models/'+model_name+'_data.pkl', 'wb'))
    print(pm.summary(trace))
    plt.show()


def infer_gss_from_model(model_name='clustering_multi_epoch_w_priors', run_name='',
                         parents={'z':None, 'w': ('z',), 'gss': ('w', 'theta')}):

    for fit_for in ['z', 'w', 'gss']:
        print('Fitting for', fit_for)
        data = pd.read_csv('../data/preprocessed/markers.csv')
        data_fit = {}
        if parents[fit_for] is not None:
            _, prev_trace = pickle.load(open('../data/models/'+model_name+'_'+parents[fit_for][0]+run_name+'.pkl', 'rb'))
            for var in parents[fit_for]:
                data_fit[var] = np.squeeze(generate_results.extract_mode_as_array(prev_trace, var=var, astype='array').astype(int))
            if fit_for == 'gss':
                _, prev_trace = pickle.load(open('../data/models/'+model_name+'_z'+run_name+'.pkl', 'rb'))
                data_fit['z'] = np.squeeze(generate_results.extract_mode_as_array(prev_trace, var='z', astype='array').astype(int))
        model_to_run = eval('models.'+model_name)
        model = model_to_run(data, fit_for, data_fit)

        with model:
            trace = pm.sample(tune=1000, init="adapt_diag", nuts_kwargs={'target_accept': 0.9})  # Jitter will likely break order constraints during training, so dont use
        pickle.dump((model, trace),
                    open('../data/models/'+model_name+'_'+ fit_for + run_name+'.pkl', 'wb'))
        pm.traceplot(trace)
        plt.show()


def plot_on_eeg(model_name='clustering_multi_epoch_w_priors', run_name=''):
    """Plot the data on EEG"""
    annot_map, epoch_map = pickle.load(open('../data/preprocessed/maps.pkl', 'rb'))

    model, trace = pickle.load(open('../data/models/'+model_name+'_z' + run_name + '.pkl', 'rb'))
    z = generate_results.extract_mode_as_array(trace, 'z', astype='dataframe')
    z['epoch_i'] = z.index.tolist()
    z['Epoch Num'] = z['epoch_i'].map({v: k for k, v in epoch_map.items()})


    # %% Load predicted spindle locs
    model, trace = pickle.load(open('../data/models/'+model_name+'_gss'+run_name+'.pkl', 'rb'))
    gss = generate_results.extract_mean_as_array(trace, 'gss', astype='dataframe')
    #gsd = generate_results.extract_mean_as_array(trace, 'gsd', astype='dataframe')
    # exp = generate_results.extract_mean_as_array(trace, 'r_E', astype='dataframe')
    # exp['rater_i'] = exp.index.tolist()
    # exp = exp.drop([0], axis=1)
    gss['epoch_i'] = gss[0]
    #gsd['epoch_i'] = gsd[0]
    #gsd['z_i'] = gsd[1]
    gss['z_i'] = gss[1]
    #gsd = gsd.drop([0, 1], axis=1)
    gss = gss.drop([0, 1], axis=1)
    #bayes_model = pd.merge(gss, gsd, on=['epoch_i', 'z_i'])
    bayes_model = gss
    bayes_model['Epoch Num'] = bayes_model['epoch_i'].map({v: k for k, v in epoch_map.items()})
    bayes_model = bayes_model.drop('epoch_i', axis=1)
    bayes_model = pd.merge(bayes_model, z, on='Epoch Num')

    # %% load link from raters spindles to predicted spindles
    model, trace = pickle.load(open('../data/models/'+model_name+'_w'+run_name+'.pkl', 'rb'))
    w = generate_results.extract_mode_as_array(trace, 'w', astype='dataframe')
    w['t'] = w.index.tolist()

    try: # handle when theta is not in model
        theta = generate_results.extract_mode_as_array(trace, 'theta', astype='dataframe')
        theta['t'] = w.index.tolist()
    except ValueError:
        theta = w.copy()
        theta['theta'] = 1

    raters_markers = pd.read_csv('../data/preprocessed/markers.csv')
    raters_markers['Epoch Num'] = raters_markers['epoch_i'].map({v: k for k, v in epoch_map.items()})
    raters_markers = raters_markers.drop(['spindle_i', 'epoch_i', 'epoch_rater_i', 'marker_per_r_i',
                                          'epoch_rater_i', 'Global Marker Index','Phase'], axis=1)
    raters_markers = pd.merge(pd.merge(raters_markers, theta, on='t'), w, on='t')
    #raters_markers = pd.merge(raters_markers, exp, on='rater_i')

    # %% calculate legit z_i's
    real_z_i = raters_markers.loc[raters_markers['theta']==1, ['w', 'Epoch Num']]
    real_z_i['z_i'] = real_z_i['w']
    real_z_i = real_z_i.drop('w', axis=1).drop_duplicates()
    bayes_model = pd.merge(bayes_model, real_z_i, on=['Epoch Num','z_i'], how='inner')
    bayes_model['s'] = bayes_model['gss']
    bayes_model['d'] = 0.5
    #bayes_model['d'] = bayes_model['gsd']
    bayes_model['e'] = bayes_model['s'] + bayes_model['d']
    bayes_model['rater_i'] = 'bayes_model'
    bayes_model = bayes_model.drop(['gss', 'epoch_i'], axis=1)
    #bayes_model = bayes_model.drop(['gss', 'gsd','epoch_i'], axis=1)

    # Import EEG!

    signals = pd.read_pickle('../data/raw/epoch_signals.pkl').set_index('Epoch Num')

    # GS from matlab
    gs_file = '../data/raw/gold_standard_spindle_markers_MODA.txt'
    matlab_gs = parse_data.load_gs_markers(gs_file)
    matlab_gs['theta'] = -1
    matlab_gs['w'] = -1
    matlab_gs['t'] = -1

    markers = pd.concat([matlab_gs, raters_markers], axis=0, sort=False)
    # markers = markers.set_index('Epoch Num')

    for epoch, predicted_markers in bayes_model.groupby('Epoch Num'):
        signal = signals.loc[epoch, :][0]
        markers_for_epoch = markers.loc[markers['Epoch Num'] == epoch, :]
        #markers_for_epoch = markers_for_epoch.sort_values('r_E', ascending=True)
        plot_epoch.plot_an_epoch(signal, predicted_markers.set_index('z_i'), markers_for_epoch)
        plt.show()
    plt.show(block=True)


def plot_on_single_epoch(model_name='single_epoch_model', run_name=''):
    """Plot the data on EEG"""
    annot_map, epoch_map = pickle.load(open('../data/preprocessed/maps.pkl', 'rb'))

    model, trace = pickle.load(open('../data/models/'+model_name+'_z' + run_name + '.pkl', 'rb'))
    z = generate_results.extract_mode_as_array(trace, 'z', astype='dataframe')
    z['epoch_i'] = z.index.tolist()
    z['Epoch Num'] = z['epoch_i'].map({v: k for k, v in epoch_map.items()})


    # %% Load predicted spindle locs
    model, trace = pickle.load(open('../data/models/'+model_name+'_gss'+run_name+'.pkl', 'rb'))
    gss = generate_results.extract_mean_as_array(trace, 'gss', astype='dataframe')
    #gsd = generate_results.extract_mean_as_array(trace, 'gsd', astype='dataframe')
    # exp = generate_results.extract_mean_as_array(trace, 'r_E', astype='dataframe')
    # exp['rater_i'] = exp.index.tolist()
    # exp = exp.drop([0], axis=1)
    gss['epoch_i'] = 0
    #gsd['epoch_i'] = gsd[0]
    #gsd['z_i'] = gsd[1]
    gss['z_i'] = gss[0]
    #gsd = gsd.drop([0, 1], axis=1)
    gss = gss.drop([0], axis=1)
    #bayes_model = pd.merge(gss, gsd, on=['epoch_i', 'z_i'])
    bayes_model = gss
    bayes_model['Epoch Num'] = bayes_model['epoch_i'].map({v: k for k, v in epoch_map.items()})
    bayes_model = bayes_model.drop('epoch_i', axis=1)
    bayes_model = pd.merge(bayes_model, z, on='Epoch Num')

    # %% load link from raters spindles to predicted spindles
    model, trace = pickle.load(open('../data/models/'+model_name+'_w'+run_name+'.pkl', 'rb'))
    w = generate_results.extract_mode_as_array(trace, 'w', astype='dataframe')
    w['t'] = w.index.tolist()
    w['w'] -= 1  # because 0=no mapping

    try: # handle when theta is not in model
        theta = generate_results.extract_mode_as_array(trace, 'theta', astype='dataframe')
        theta['t'] = w.index.tolist()
    except ValueError:
        theta = w.copy()
        theta['theta'] = 1

    raters_markers = pd.read_csv('../data/preprocessed/markers.csv')
    raters_markers['Epoch Num'] = raters_markers['epoch_i'].map({v: k for k, v in epoch_map.items()})
    raters_markers = raters_markers.drop(['spindle_i', 'epoch_i', 'epoch_rater_i', 'marker_per_r_i',
                                          'epoch_rater_i', 'Global Marker Index','Phase'], axis=1)
    raters_markers = pd.merge(pd.merge(raters_markers, theta, on='t'), w, on='t')
    #raters_markers = pd.merge(raters_markers, exp, on='rater_i')

    # %% calculate legit z_i's
    real_z_i = raters_markers.loc[raters_markers['theta']==1, ['w', 'Epoch Num']]
    real_z_i['z_i'] = real_z_i['w']
    real_z_i = real_z_i.drop('w', axis=1).drop_duplicates()
    bayes_model = pd.merge(bayes_model, real_z_i, on=['Epoch Num','z_i'], how='inner')
    bayes_model['s'] = bayes_model['gss']
    bayes_model['d'] = 0.5
    #bayes_model['d'] = bayes_model['gsd']
    bayes_model['e'] = bayes_model['s'] + bayes_model['d']
    bayes_model['rater_i'] = 'bayes_model'
    bayes_model = bayes_model.drop(['gss', 'epoch_i'], axis=1)
    #bayes_model = bayes_model.drop(['gss', 'gsd','epoch_i'], axis=1)

    # Import EEG!

    signals = pd.read_pickle('../data/raw/epoch_signals.pkl').set_index('Epoch Num')

    # GS from matlab
    gs_file = '../data/raw/gold_standard_spindle_markers_MODA.txt'
    matlab_gs = parse_data.load_gs_markers(gs_file)
    matlab_gs['theta'] = -1
    matlab_gs['w'] = -1
    matlab_gs['t'] = -1

    markers = pd.concat([matlab_gs, raters_markers], axis=0, sort=False)
    # markers = markers.set_index('Epoch Num')

    for epoch, predicted_markers in bayes_model.groupby('Epoch Num'):
        signal = signals.loc[epoch, :][0]
        markers_for_epoch = markers.loc[markers['Epoch Num'] == epoch, :]
        #markers_for_epoch = markers_for_epoch.sort_values('r_E', ascending=True)
        plot_epoch.plot_an_epoch(signal, predicted_markers.set_index('z_i'), markers_for_epoch)
        plt.show()
    plt.show(block=True)

def plot_on_single_epoch(model_name='single_epoch_model', run_name=''):
    """Plot the data on EEG"""
    annot_map, epoch_map = pickle.load(open('../data/preprocessed/maps.pkl', 'rb'))

    model, trace = pickle.load(open('../data/models/'+model_name+'_z' + run_name + '.pkl', 'rb'))
    z = generate_results.extract_mode_as_array(trace, 'z', astype='dataframe')
    z['epoch_i'] = z.index.tolist()
    z['Epoch Num'] = z['epoch_i'].map({v: k for k, v in epoch_map.items()})


    # %% Load predicted spindle locs
    model, trace = pickle.load(open('../data/models/'+model_name+'_gss'+run_name+'.pkl', 'rb'))
    gss = generate_results.extract_mean_as_array(trace, 'gss', astype='dataframe')
    #gsd = generate_results.extract_mean_as_array(trace, 'gsd', astype='dataframe')
    # exp = generate_results.extract_mean_as_array(trace, 'r_E', astype='dataframe')
    # exp['rater_i'] = exp.index.tolist()
    # exp = exp.drop([0], axis=1)
    gss['epoch_i'] = 0
    #gsd['epoch_i'] = gsd[0]
    #gsd['z_i'] = gsd[1]
    gss['z_i'] = gss[0]
    #gsd = gsd.drop([0, 1], axis=1)
    gss = gss.drop([0], axis=1)
    #bayes_model = pd.merge(gss, gsd, on=['epoch_i', 'z_i'])
    bayes_model = gss
    bayes_model['Epoch Num'] = bayes_model['epoch_i'].map({v: k for k, v in epoch_map.items()})
    bayes_model = bayes_model.drop('epoch_i', axis=1)
    bayes_model = pd.merge(bayes_model, z, on='Epoch Num')

    # %% load link from raters spindles to predicted spindles
    model, trace = pickle.load(open('../data/models/'+model_name+'_w'+run_name+'.pkl', 'rb'))
    w = generate_results.extract_mode_as_array(trace, 'w', astype='dataframe')
    w['t'] = w.index.tolist()
    w['w'] -= 1  # because 0=no mapping

    try: # handle when theta is not in model
        theta = generate_results.extract_mode_as_array(trace, 'theta', astype='dataframe')
        theta['t'] = w.index.tolist()
    except KeyError:
        theta = w.copy()
        theta['theta'] = ~(theta['w'] == -1)
        theta = theta.drop('w', axis=1)

    raters_markers = pd.read_csv('../data/preprocessed/markers.csv')
    raters_markers['Epoch Num'] = raters_markers['epoch_i'].map({v: k for k, v in epoch_map.items()})
    raters_markers = raters_markers.drop(['spindle_i', 'epoch_i', 'epoch_rater_i', 'marker_per_r_i',
                                          'epoch_rater_i', 'Global Marker Index','Phase'], axis=1)
    raters_markers = pd.merge(pd.merge(raters_markers, theta, on='t'), w, on='t')
    #raters_markers = pd.merge(raters_markers, exp, on='rater_i')

    # %% calculate legit z_i's
    real_z_i = raters_markers.loc[raters_markers['theta']==1, ['w', 'Epoch Num']]
    real_z_i['z_i'] = real_z_i['w']
    real_z_i = real_z_i.drop('w', axis=1).drop_duplicates()
    bayes_model = pd.merge(bayes_model, real_z_i, on=['Epoch Num','z_i'], how='inner')
    bayes_model['s'] = bayes_model['gss']
    bayes_model['d'] = 0.5
    #bayes_model['d'] = bayes_model['gsd']
    bayes_model['e'] = bayes_model['s'] + bayes_model['d']
    bayes_model['rater_i'] = 'bayes_model'
    bayes_model = bayes_model.drop(['gss', 'epoch_i'], axis=1)
    #bayes_model = bayes_model.drop(['gss', 'gsd','epoch_i'], axis=1)

    # Import EEG!

    signals = pd.read_pickle('../data/raw/epoch_signals.pkl').set_index('Epoch Num')

    # GS from matlab
    gs_file = '../data/raw/gold_standard_spindle_markers_MODA.txt'
    matlab_gs = parse_data.load_gs_markers(gs_file)
    matlab_gs['theta'] = -1
    matlab_gs['w'] = -1
    matlab_gs['t'] = -1

    markers = pd.concat([matlab_gs, raters_markers], axis=0, sort=False)
    # markers = markers.set_index('Epoch Num')

    for epoch, predicted_markers in bayes_model.groupby('Epoch Num'):
        signal = signals.loc[epoch, :][0]
        markers_for_epoch = markers.loc[markers['Epoch Num'] == epoch, :]
        #markers_for_epoch = markers_for_epoch.sort_values('r_E', ascending=True)
        plot_epoch.plot_an_epoch(signal, predicted_markers.set_index('z_i'), markers_for_epoch)
        plt.show()
    plt.show(block=True)

def plot_wald():
    spindle_refractory_mu = 8.81
    spindle_refractory_lam = 14.91
    x = np.linspace(0.1, 40, 2000)
    wald = np.vectorize(lambda x: models.wald_refractory_diff_contraint_np(x, spindle_refractory_mu, spindle_refractory_lam))
    y = wald(x)
    print(max(y))
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    # print('running')
    # models.test_switching()
    # sys.exit(0)
    #create_and_save_dataset()
    #plot_wald()
    infer_gss_from_model(model_name='single_epoch_simple', parents={'z':None, 'w': ('z',), 'gss': ('w')})
    #fitting_priors.fit_spindle_density_prior()
    #fitting_priors.fit_spindle_duration()
    #fitting_priors.get_samples_for_refractory()
    #fitting_priors.fit_refractory_minus_duration()
    #models.plot_my_fun()
    #create_and_save_dataset(cluster=False)
    #plot_on_eeg(model_name='single_epoch_simple')
    plot_on_single_epoch(model_name='single_epoch_simple')
    #infer_gss_from_model()
