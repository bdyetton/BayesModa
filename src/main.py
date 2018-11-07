import matplotlib
matplotlib.use('TkAgg')
from src import parse_data, models, generate_results, fitting_priors
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np


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


def infer_gss_from_model():
    #%% Get Z
    parents = {'z':None, 'w': ('z',), 'gss': ('w', 'theta')}

    for fit_for in ['z', 'w', 'gss']:
        print('Fitting for', fit_for)
        data = pd.read_csv('../data/preprocessed/markers.csv')
        data_fit = {}
        if parents[fit_for] is not None:
            _, prev_trace = pickle.load(open('../data/models/clustering_multi_epoch_w_priors_'+parents[fit_for][0]+'.pkl', 'rb'))
            for var in parents[fit_for]:
                data_fit[var] = np.squeeze(generate_results.extract_mode_as_array(prev_trace, var=var)).astype(int)
            if fit_for == 'gss':
                _, prev_trace = pickle.load(open('../data/models/clustering_multi_epoch_w_priors_z.pkl', 'rb'))
                data_fit['z'] = np.squeeze(generate_results.extract_mode_as_array(prev_trace, var='z')).astype(int)
        model = models.clustering_multi_epoch_w_priors(data, fit_for, data_fit)

        with model:
            trace = pm.sample(init="adapt_diag", nuts_kwargs={'target_accept': 0.99})  # Jitter will likely break order constraints during training, so dont use
        pickle.dump((model, trace),
                    open('../data/models/clustering_multi_epoch_w_priors_' + fit_for + '.pkl', 'wb'))
        pm.traceplot(trace)
        plt.show()


def plot_on_eeg():
    """Plot the data on EEG"""
    model, trace = pickle.load(open('../data/models/clustering_multi_epoch_w_priors_gss.pkl', 'rb'))
    gss = generate_results.extract_mean_as_array(trace, 'gss', astype='dataframe')
    gsd = generate_results.extract_mean_as_array(trace, 'gsd', astype='dataframe')
    gss['epoch_i'] = gss[0]
    gsd['epoch_i'] = gsd[0]
    gsd['z_i'] = gsd[1]
    gss['z_i'] = gss[1]
    gsd = gsd.drop([0, 1], axis=1)
    gss = gss.drop([0, 1], axis=1)

    model, trace = pickle.load(open('../data/models/clustering_multi_epoch_w_priors_z.pkl', 'rb'))
    z = pd.DataFrame(generate_results.extract_mode_as_array(trace, 'z').astype(int), columns=['z'])
    z['epoch_i'] = z.index.tolist()

    epoch_map = pickle.load(open('../data/preprocessed/maps.pkl', 'rb'))[2]
    gsd['Epoch Num'] = gsd['epoch_i'].map({v:k for k,v in epoch_map.items()})
    model_gs = pd.merge(gss, gsd, on=['epoch_i', 'z_i'], how='left')
    model_gs = pd.merge(model_gs, z, on='epoch_i', how='left')
    model_gs = model_gs.loc[model_gs['z_i'] < model_gs['z']]

    # Import EEG!
    signals = pd.read_pickle('../data/raw/epoch_signals.pkl').set_index('Epoch Num')

    # GS from matlab
    gs_file = '../data/raw/gold_standard_spindle_markers_MODA.txt'
    matlab_gs = parse_data.load_gs_markers(gs_file).set_index('Epoch Num')

    for epoch, epoch_data in model_gs.groupby('Epoch Num'):
        plt.figure(figsize=[10, 2])
        try:
            matlab_gs_i = matlab_gs.loc[[epoch], :]
        except KeyError:
            pass
        else:
            gs_s = matlab_gs_i['gss']
            gs_e = matlab_gs_i['gss'] + matlab_gs_i['gsd']
            for x, y in zip(gs_s, gs_e):
                plt.axvspan(x, y, color='yellow', alpha=0.5)

        for idx, row in epoch_data.iterrows():
            plt.axvspan(row['gss'], row['gss']+row['gsd'], color='red', alpha=0.5)

        #plot signal
        signal = signals.loc[epoch, :][0]
        times = list(np.linspace(0, 25, len(signal)))
        plt.plot(times, signal)
        plt.show(block=True)


def plot_wald():
    spindle_refractory_mu = 8.81
    spindle_refractory_lam = 14.91
    x = np.linspace(0.1, 40, 2000)
    wald = np.vectorize(lambda x: models.wald_refractory_diff_contraint_np(x, spindle_refractory_mu, spindle_refractory_lam))
    y = wald(x)
    print(max(y))
    plt.plot(x,y)
    plt.show()


if __name__ == "__main__":
    #create_and_save_dataset()
    #plot_wald()
    infer_gss_from_model()
    #fitting_priors.fit_spindle_density_prior()
    #fitting_priors.fit_spindle_duration()
    #fitting_priors.get_samples_for_refractory()
    #fitting_priors.fit_refractory_minus_duration()
    #models.plot_my_fun()
    #create_and_save_dataset(cluster=False)
    plot_on_eeg()
    #infer_gss_from_model()
