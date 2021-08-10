import matplotlib.pyplot as plt
import numpy as np
# import scipy.ndimage
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gsmooth
from scipy.optimize import curve_fit

data_file = r"C:\Users\Leo\Documents\TwoDrive\Honours\Data\data_dump/210609_T1_7_raw.txt"

data = pd.read_csv(data_file, skiprows=20, names=['time', 'counts'], delimiter='\t')

print("Smoothing data")
data['smooth'] = gsmooth(data.counts, sigma=5)
# plt.plot(data.time, data.smooth)
# plt.show()

max_counts = max(data.counts)
thresh = 0.8 * max_counts

def get_rise_falls(_data, thresh):
    """ Return indexes of rises and falls as two arrays"""
    triggered_points = (_data > thresh).astype(int)
    cross_overs = triggered_points[1:] - triggered_points[:-1]
    rise_points = cross_overs > 0
    rise_points = np.flatnonzero(rise_points)

    fall_points = cross_overs < 0
    fall_points = np.flatnonzero(fall_points)

    return rise_points, fall_points

rises, falls = get_rise_falls(np.array(data.smooth), thresh)
rise_ts, fall_ts = data.time[rises], data.time[falls]

plt.figure()
plt.plot(data.time / 1e3, data.counts)
plt.title("Counts with rise and fall indicators")
plt.xlabel("Time [us]")
plt.ylabel("Counts")
plt.vlines(rise_ts/1e3, 0, max_counts*1.1, colors='green')
plt.vlines(fall_ts/1e3, 0, max_counts*1.1, colors='red')

def get_readout_start_and_ends(rise_ts, fall_ts):
    """ Assumes that the first rise and fall are always 
    from the same readout pulse. """
    fin_len = min(len(rise_ts), len(fall_ts))
    result = np.zeros((fin_len, 2))
    result[:,0] = rise_ts[:fin_len]
    result[:,1] = fall_ts[:fin_len]
    return result

pulses = get_readout_start_and_ends(rise_ts, fall_ts)
pulses_idx = get_readout_start_and_ends(rises, falls)

print(f"Average Pulse length: {np.diff(np.mean(pulses, axis=0))[0]:.2f} "
                             f"+/- {np.std(np.diff(pulses, axis=1)):.2f} ns")    

integration_width = 40 # time steps, each step is 3.2 ns
# readout_counts = np.array(data.counts)[[pulses[:, 0]: pulses[:, 0] + integration_width]] 
forward = np.arange(integration_width)
forward = np.tile(forward, (len(pulses), 1))
backward = -forward

reads = (forward + pulses_idx[:, [0]]).astype(int)
refs = (backward + pulses_idx[:, [1]]).astype(int)


arr = np.array(data.smooth)
reads = np.sum(arr[reads], axis=-1)
refs = np.sum(arr[refs], axis=-1)

# Fit to exponential
decay_y = (reads[1:] / refs[:-1])
decay_x = pulses[1:, 0] - pulses[:-1, 1]
decay_x_idx = decay_x / (data.time[1] - data.time[0])
noise_region_idx = pulses_idx[:-1, 1] + decay_x_idx // 2
noise_regions = forward[:-1] + noise_region_idx.reshape((-1, 1))
noise_regions = arr[noise_regions.astype(int)]
noise_avgs = np.mean(noise_regions, axis=-1)

def decay_f(x, T1, a, base):
    return a * np.exp(-x/T1) + base
fit_args = {}
fit_args.update(p0=(2e5, 1, 0), bounds=[(0, 0, 0), (1e6, np.inf, np.inf)])
popt, pcov = curve_fit(decay_f, decay_x, decay_y, **fit_args)

T1_val = popt[0] / 1e3 # us
T1_err = np.sqrt(pcov[0,0]) / 1e3 # us

print(f"T1: {T1_val:.2f} +/- {T1_err:.2f} us")

plt.figure()

x_sort = np.argsort(decay_x)
plt.plot(decay_x[x_sort], decay_y[x_sort], "--.", label="Data")
plt.plot(data.time[::10], decay_f(data.time[::10], *popt), label=f"Fit: T1 = {T1_val:.2f} +/- {T1_err:.2f} us")
plt.legend()
plt.grid()
plt.title("T1 Decay Data with fit")
plt.xlabel("Time [ns]")
plt.ylabel("PL [arb.]")

# High / Low noise:
plt.figure()
noise_thresh = 0.5 * max(noise_avgs)
high_noise = noise_avgs > noise_thresh

popt_hn, pcov_hn = curve_fit(decay_f, decay_x[high_noise], decay_y[high_noise], **fit_args)
popt_ln, pcov_ln = curve_fit(decay_f, decay_x[~high_noise], decay_y[~high_noise], **fit_args)
T1_hn = popt_hn[0] / 1e3
T1_err_hn = np.sqrt(pcov_hn[0,0]) / 1e3
T1_ln = popt_ln[0] / 1e3
T1_err_ln = np.sqrt(pcov_ln[0,0]) / 1e3

fit_x = data.time[::10]

plt.plot(fit_x, decay_f(fit_x, *popt_hn), label="Fit to high noise, T1 = "f"{T1_hn:.2f} +/- {T1_err_hn:.2f} us")
plt.plot(fit_x, decay_f(fit_x, *popt_ln), label="Fit to low noise, T1 = "f"{T1_ln:.2f} +/- {T1_err_ln:.2f} us")

plt.plot(decay_x[x_sort][high_noise], decay_y[x_sort][high_noise], "--.", label="High noise data")
plt.plot(decay_x[x_sort][~high_noise], decay_y[x_sort][~high_noise], "--.", label="Low noise data")
plt.legend()
plt.grid()
plt.title("T1 Decay Data separated into noisy and not noisy")
plt.xlabel("Time [ns]")
plt.ylabel("PL [arb.]")

# step = 2
# plt.figure()
# plt.plot(pulses[2::step, 0] - pulses[:-2:step, 1], reads[2::step]/refs[2::step], '.')
# plt.plot(pulses[1::step, 0] - pulses[:-1:step, 1], reads[1::step]/refs[1::step], '.')
# plt.title("Odd and even pulses")

# Plot good pulses on top of eachother
good_pulses = pulses_idx[1:][~high_noise]
pulse_shapes = [data.counts[slice(*p)] for p in good_pulses.astype(int)]

plt.figure()
for i, p in enumerate(pulse_shapes):
    p_len = len(p)
    plt.plot(data.time[:p_len], p, label=f"{i}")

plt.xlabel("Time [ns]")
plt.ylabel("Counts [per ns]")
plt.legend()
plt.grid()
plt.show()
