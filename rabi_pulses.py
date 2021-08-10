from pbwx import pulse_utils as pu#, load_pulse as lp
from astropy import units as u
import numpy as np

# load_pulse.read_pulse_file()

MCS_BIT = 3
IR_BIT = 4
GREEN_BIT = 5

excite_pulse = 3*u.us

prog = pu.SequenceProgram()
full_seq = pu.PulseSequence(prog)

start = pu.RawSequence(full_seq.controller)
start.add_seq(MCS_BIT, [0, 10*u.ns,])
start_mark = start.get_marker(0)
tau_start = 12 * u.ns
tau_end   = 500 * u.ns
green_seq = [0]
for tau in np.linspace(tau_start, tau_end, 20):
    green_seq.append(excite_pulse)
    green_seq.append(tau)

start.add_seq(GREEN_BIT, green_seq)

start.plot_sequence()



