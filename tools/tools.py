import numpy as np

def find_peaks_sp(data, num_peaks):
    """ Find a specific number of peaks by finding the stationary points with the
    highest values in data. """
    diff = np.diff(data)
    # Find which values cross over zero
    prod = diff[1:] * diff[:-1]
    # prod will be negative where diff has crossed the x-axis
    sps = np.flatnonzero(prod < 0) + 1

    values = data[sps]
    pks = sps[values.argsort()[::-1][:num_peaks]]

    return pks

def get_rise_falls(_data, thresh):
    """ Return indexes of rises and falls as two arrays"""
    # print(f"Max in data: {max(_data)}, thresh: {thresh}")
    triggered_points = (_data > thresh).astype(int)
    cross_overs = triggered_points[1:] - triggered_points[:-1]
    rise_points = cross_overs > 0
    rise_points = np.flatnonzero(rise_points)

    fall_points = cross_overs < 0
    fall_points = np.flatnonzero(fall_points)

    return rise_points, fall_points