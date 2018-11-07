import pandas as pd
import numpy as np


def is_overlaping(start1: float, end1: float, start2: float, end2: float) -> float:
    """how much does the range (start1, end1) overlap with (start2, end2)
    Looks strange, but algorithm is tight and tested.

    Args_i:
        start1: start of interval 1, in any unit
        end1: end of interval 1
        start2: start of interval 2
        end2: end of interval 2

    Returns:
        overlap of intervals in same units as supplied."""
    return max(max((end2 - start1), 0) - max((end2 - end1), 0) - max((start2 - start1), 0), 0)


def load_and_parse_for_modeling(marker_file):
    markers, markers_og = load_markers(marker_file)
    unique_annots = markers['annotatorID'].unique()
    annot_map = {v: k for k, v in enumerate(unique_annots)}
    epoch_i_map = {v: k for k, v in enumerate(markers['epoch_i'].unique())}
    markers['epoch_i'] = markers['epoch_i'].map(epoch_i_map)
    markers['rater_i'] = markers_og['annotatorID'].map(annot_map)
    markers = markers.dropna()  # Because some raters are unknown, "other" type, which will apear as NA. FIXME Maybe we should code them soemthing different?
    markers['rater_i'] = markers['rater_i'].astype(int)
    markers['spindle_i'] = markers.groupby(['epoch_i', 'rater_i']).cumcount()
    markers['marker_per_r_i'] = markers.groupby(['epoch_i', 'rater_i'])['Phase'].transform('count')
    markers['conf'] = markers['Conf'].map({'low':0.25, 'med':0.5, 'high':0.99})
    markers['epoch_rater_i'] = markers.groupby(['epoch_i', 'rater_i']).ngroup()
    markers = markers.drop('Conf',axis=1).dropna()
    markers['t'] = list(range(markers.shape[0]))
    return markers, annot_map, epoch_i_map


def load_markers(marker_file):
    markers_og = pd.read_csv(marker_file, delimiter='\t')
    markers_og = markers_og.dropna()
    markers = markers_og.loc[:, ['Phase', 'Global Marker Index', 'annotatorID', 'Conf']]
    markers['s'] = markers_og['MASS Marker Start Time (s)'] - markers_og['MASS Epoch Start Time (s)']
    markers['d'] = markers_og['MASS Marker End Time (s)'] - markers_og['MASS Marker Start Time (s)']
    markers['e'] = markers_og['MASS Marker End Time (s)'] - markers_og['MASS Epoch Start Time (s)']
    markers['epoch_i'] = markers_og['Epoch Num']
    return markers, markers_og


def load_gs_markers(marker_file):
    markers_og = pd.read_csv(marker_file, delimiter='\t')
    markers_og = markers_og.dropna()
    markers = markers_og.loc[:, ['Phase', 'Global Marker Index', 'annotatorID']]
    markers['gss'] = markers_og['MASS Marker Start Time (s)'] - markers_og['MASS Epoch Start Time (s)']
    markers['gsd'] = markers_og['MASS Marker End Time (s)'] - markers_og['MASS Marker Start Time (s)']
    markers['Epoch Num'] = markers_og['Epoch Num']
    markers['Type'] = 'matlab'
    return markers


def cluster_by_overlap(markers, start_var='s', end_var='e'):
    raise DeprecationWarning('This is not a good way to cluster')
    markers['cluster'] = np.nan #FIXME this should be cluster_num
    marker_cont = []
    cluster_counter = -1
    for epoch, epoch_markers in markers.groupby('Epoch Num'):
        cluster_counter += 1
        epoch_markers = epoch_markers.sort_values(start_var).reset_index(drop=True)
        n_markers_in_epoch = epoch_markers.shape[0]
        epoch_markers.loc[0, 'cluster'] = cluster_counter  # first marker of an epoch always gets a new gs_i
        for idx1 in range(0, n_markers_in_epoch-1): # check through all markers in epoch and see if they overlap.
            idx2 = idx1+1
            marker1 = epoch_markers.loc[idx1, :]
            marker2 = epoch_markers.loc[idx2, :]
            if not is_overlaping(marker1[start_var], marker1[end_var], marker2[start_var], marker2[end_var]): # If they overlap, assign the same gs_i, else new gs_i needed
                cluster_counter += 1  # new gs_i marker
            epoch_markers.loc[idx2, 'cluster'] = cluster_counter
        marker_cont.append(epoch_markers)
    markers = pd.concat(marker_cont, axis=0)
    markers['cluster'] = markers['cluster'].astype(int)
    return markers
