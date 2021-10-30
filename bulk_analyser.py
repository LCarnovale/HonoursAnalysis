import os
import sys

import ipywidgets as wwid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from scipy.ndimage import gaussian_filter, gaussian_filter1d as g_filter

from scipy.optimize import curve_fit

import tools.file_finder
import tools.helpers as helpers
from tools.file_finder import KeyDoesNotExistError, find_file, AmbiguousSpecificationError

options = [
    "odmr-2dgrid",
    "odmr-raw",
    "odmr-smooth",
    "rabi-signal",
    "rabi-fit",
]

output_root = "./"

try:
    selected = sys.argv[1]
except IndexError:
    selected = "odmr-2dgrid"
else:
    if selected not in options:
        raise ValueError(f"{selected} is not a valid option.\n"
                         f"Valid options are: {options}")

try:
    root = output_root + selected + "/"
    if not os.path.exists(root):
        os.makedirs(root)
except OSError:
    print("Unable to create directory %s" % root)
    exit()

START_DATE = (2021, 5, 13)
END_DATE = (2021, 11, 19)
MAX_N = 200

START_DATE = "{:d}-{:02d}-{:02d}".format(*START_DATE)
END_DATE = "{:d}-{:02d}-{:02d}".format(*END_DATE)

def reprint(text):
    sys.stdout.write("\033[K")
    print(text, end='\r', flush=True)

files = []
names = []
kwargs = {}
if "odmr" in selected:
    arg = "ODMR"
    kwargs.update(contains="Counts", verbose=False, 
        cache=False, return_offline=False)
elif "rabi" in selected:
    arg = "NV"
    kwargs.update(raw=True, verbose=False, 
        cache=False, return_offline=False)

for date in helpers.date_range(START_DATE, END_DATE):
    # if date.weekday() in (2, 3, 5, 6):
    #     continue
    date_str = date.strftime("%y%m%d")
    num_found = 0
    biggest_n = 1
    try:
        for n in range(1, MAX_N + 1):
            try:

                files.append(find_file(arg, n, date_str, **kwargs))
            except KeyDoesNotExistError as e:
                if not e.date_exists:
                    # Skip this date
                    break
            except FileNotFoundError as e:
                pass
            except AmbiguousSpecificationError:
                print("Ambiguous specs for %s: %s" % (date_str, n))
            except KeyboardInterrupt as e:
                raise e
            else:
                num_found += 1
                names.append(f"{date_str}-{n}")
                if num_found > 1:
                    reprint(f"{date_str}: " + "."*num_found)
        if n > biggest_n: biggest_n = n
        if num_found > 0:
            reprint(f"{date_str}: {num_found}")
            print()
    except KeyboardInterrupt:
        break

print("Found %d %s files" % (len(files), arg))
# All files found
    
    # Make ODMR grids
try:
    first_one = True
    for file, name in zip(files, names):
        try:
            if selected == "odmr-2dgrid":
                title = f"ODMR Sweep Signals {name}"
                full_spec = np.genfromtxt(file)
                freq_ax, counts = full_spec[0], full_spec[1:]
                # Normalise each sweep
                counts /= counts.max(axis=1, keepdims=True)
                fig = plt.figure(figsize=(10, 10))
                ax = fig.subplots(1, 1)
                ax.set_title(title)
                m = ax.matshow(gaussian_filter(counts, (1, 1)), cmap="viridis")
                plt.colorbar(m)
                ax.set_xlabel("Frequency (MHz)")
                ax.set_ylabel("Sweep number")
                fig.savefig(root + name + ".png", )

            elif selected == "rabi-fit":
                title = f"Rabi signal fit {name}"
                rabi_spec = pd.read_csv(file, skiprows=20, names=['time', 'counts'], delimiter='\t')
                freq, powr = helpers.get_val_from_file(f, "RF freq.", "RF power")

            
            if first_one:
                plt.show()
                continue_ = input("Continue? (Y/n) ")
                if continue_ == "n":
                    break
                first_one = False
            else:
                plt.close()
        except ValueError as e:
            print(e)
            print("Skipping %s" % name)
except OSError:
    print("Could not create directory %s" % selected)
except KeyboardInterrupt:
    print("Aborting")
    exit()
except Exception as e:
    print("Error, must exit.")
    raise e

