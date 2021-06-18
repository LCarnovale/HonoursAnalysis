from pbwx import pulse_utils as pu#, load_pulse as lp
from astropy import units as u

# load_pulse.read_pulse_file()

prog = pu.SequenceProgram()
full_seq = pu.PulseSequence(prog)
