import sys
import astropy
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from numpy import random
import scipy

# Simulate PL results from a Rabi measurement

##### \/ \/ Parameters \/ \/ #####
RABI_PI_TIME = 800 * u.ns               # nanoseconds
BGRD_B_FIELD = 0.1 * u.mT               # milli-Teslas
MAX_CONTRAST_PL = 0.9                   # percentage
PHASE_COHERENCE = 3 * u.us              # Lifetime of coherence of Rabi oscillations 
                                        # (ie decay lifetime of oscillations)
TIME_RES = 3. * u.ns                    # Time step / Time resolution
EXCITE_RISE = 50*u.ns                # Rise time of the excitation pulse
EXCITE_FALL = 50*u.ns                # Fall time of the excitation pulse
EXCITE_TIME = 3*u.us                   # Excitation laser duration
TAU_START = 50 * u.ns
TAU_END   = 4000 * u.ns
NUM_POINTS = 60                        # Number of measurements spread between TAU_START and TAU_END
# MAX_COUNTS = 1000 / u.ms                # counts per millisecond 
BGRD_COUNTS = 1200 / u.us                 # background counts per millisecond

COUNTS_NOISE = 0.01                     # parts per thousand
NV_COUNT = 10000                         # Number of NVs to consider. Each one will either be bright or dark.
DARK_PL_PROB = 0.9                      # Probability of PL from the dark state
BRIGHT_PL_PROB = 0.99                   # Probability of PL from the bright state  
NV_CYCLE_TIME = 15 * u.ns               # Duration of cycle of NV through ground->excited->ground transitions.

LEADING_WAIT_TIME = 500 * u.ns          # Delay before the first pulse
##### /\ /\ Parameters /\ /\#####

NONE = -1
BRIGHT = 1
DARK = 0
D2B_PROB = 1 - DARK_PL_PROB # Dark to bright
B2D_PROB = 1 - BRIGHT_PL_PROB # Bright to dark


COUNTS_NOISE /= 1e3
omega_rabi = np.pi * u.rad / RABI_PI_TIME

def FLIP(nv):
    temp = nv.copy()
    temp[nv == BRIGHT] = DARK
    temp[nv == DARK] = BRIGHT
    return temp

def rabi_flop(t, omega_rabi,):
    """ Return the probability of observing a transition after time `t`
    with a driving field of frequency `omega_drive`, and a Rabi frequency
    of `omega_rabi`. 
    
    All inputs should have appropriate units (`t`:seconds, `omega_xyz`:per seconds)"""

    # Omega = np.sqrt((omega_drive - omega_field)**2 + omega_field**2)
    flip_p = np.sin(omega_rabi * t / 2)**2
    random_flip = np.ones(np.shape(t)) / 2
    decay_factor = np.exp(-t / PHASE_COHERENCE)
    # Give a weighted average
    return decay_factor * flip_p + random_flip * (1 - decay_factor)

def NV_cycle(NVs, cycles):
    """ Determine the result of cycling all NVs over a given number
    cycles. Returns a new array with the final state of each NV"""
    # Get probability of NVs going to respective states after being excited
    new_nvs = np.full(NVs.shape, DARK)
    while cycles > 0:
        dark_probs = np.zeros(NVs.shape)
        bright_probs = np.zeros(NVs.shape)
        bright_nvs = NVs == BRIGHT
        dark_nvs = NVs == DARK
        bright_probs[bright_nvs] = BRIGHT_PL_PROB
        bright_probs[dark_nvs] = D2B_PROB
        random_nums = random.random(NVs.shape)
        # Which NVs end up bright:
        new_nvs[bright_probs > random_nums] = BRIGHT
        # new_nvs[dark_probs < random_nums] = DARK
        cycles -= 1
    return new_nvs

def excite(NVs, time, rise_t=EXCITE_RISE, fall_t=EXCITE_FALL):
    """ Simulates exciting NVs by cycling them, measures
    PL. Returns the final NV states and PL as an array for
    each cycle. 
    
    Adds extra time for fall time of the excitation laser.
    
    Results will have time resolution of NV_CYCLE_TIME"""
    time += fall_t
    cycles = time // NV_CYCLE_TIME + 0
    old = NVs.copy()
    PL = []
    t = 0
    while cycles > 0:
        new = NV_cycle(old, 1)
        t += NV_CYCLE_TIME
        # PL comes from NVs that didn't go via non radiative paths
        pl_nvs = new == old
        pl_count = pl_nvs.sum()
        if rise_t != 0:
            if t < rise_t:
                f = t / rise_t
                pl_count *= f
        if fall_t != 0:
            if time - t < fall_t:
                f = (time - t) / fall_t
                pl_count *= f
        PL.append(pl_count + 0)
        old = new
        cycles -= 1

    return new, PL

def get_noise(t):
    steps = int(t // TIME_RES)
    total_counts = t * BGRD_COUNTS
    counts = np.full(steps, total_counts / steps)
    noise = random.normal(steps) * COUNTS_NOISE * counts
    return noise

# Prepare NVs and polarise into bright state
dark_nvs = np.full(NV_COUNT, DARK)
nv_array = NV_cycle(dark_nvs, 60)


# dark_nvs = np.full(NV_COUNT, DARK)
# x = NV_cycle(dark_nvs, 1)

signal = []
t_now = 0.
def add_signal(data, time_res=TIME_RES):
    global t_now
    time_full = len(data) * time_res
    if time_res != TIME_RES:
        recorded_time_steps = int(time_full // TIME_RES)
        new_time = np.arange(recorded_time_steps) * TIME_RES
        old_time = np.arange(len(data)) * time_res
        new_time = new_time[new_time < max(old_time)]
        new_data = np.interp(new_time, old_time, data)
        data = new_data
    t_now += time_full
    signal.append(data)

# Wait
pre_noise = get_noise(LEADING_WAIT_TIME)
add_signal(pre_noise)

# Excite
nvs, PL = excite(nv_array, EXCITE_TIME)
add_signal(PL, time_res=NV_CYCLE_TIME)

tau_vals = np.linspace(TAU_START, TAU_END, NUM_POINTS)
# tau = tau_vals[0]
for tau in tau_vals:
    print(f"tau: {tau:.2f} / {TAU_END:.2f}", end='\r', flush=True)
    # Drive for time tau
    flip_prob = rabi_flop(tau, omega_rabi)
    rand_n = random.random(nvs.shape)
    # Flip if necessary
    # print(f"flip prob: {flip_prob}, flipping {(rand_n < flip_prob).sum()} nvs")
    nvs[rand_n < flip_prob] = FLIP(nvs[rand_n < flip_prob])
    # Add noise while we drove NVs
    add_signal(get_noise(tau))
    # Excite NVs
    nvs, pl = excite(nvs, EXCITE_TIME)
    add_signal(pl, NV_CYCLE_TIME)
print()
final_signal = np.concatenate(signal)
t_ax = np.arange(final_signal.size) * TIME_RES

np.savetxt("sim_out.txt", np.array([t_ax, final_signal]).T, fmt="%.2f")
print("Saved.")

if "--plot" in sys.argv:
    plt.plot(t_ax.to("ns"), final_signal)
    plt.xlabel(f"t [{t_ax.unit}]")
    plt.show()

    
    # 
# if __name__ == "__main__":
    # break