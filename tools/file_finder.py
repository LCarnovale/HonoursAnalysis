import os
import sys
import threading
import shutil
import re

DUMP_PATH = r"C:\Users\Leo\Documents\TwoDrive\Honours\Data\data_dump/"
NETWORK_PATH = r"U:\gopto\OPTOSYNC\Tweezers1/"

EXTRA_PATHS = [r"U:\gopto\OPTODOCS\1Leo/"]

class AmbiguousSpecificationError(Exception):
    def __init__(self, *args, **kwargs):
        super(AmbiguousSpecificationError, self).__init__(*args, **kwargs)

class KeyDoesNotExistError(FileNotFoundError):
    def __init__(self, *args, date_exists=None, n_exists=None, **kwargs):
        if date_exists and n_exists:
            raise ValueError("Both key and n can not exist for this exception to be raised.")
        self.date_exists = date_exists
        self.n_exists = n_exists
        super(KeyDoesNotExistError, self).__init__(*args, **kwargs)

def get_offline_name(exp_type, num, date, raw, contains=None):
    return (f"{date}_{exp_type}_{num}" +
        ("_" + contains + "_" if contains else "") +
        ("_raw" if raw else "") + ".txt")
        

def find_file(exp_type: str, num: int, date: str, raw=False, cache=True, 
        return_offline=True, contains:str="", verbose=True) -> str:
    """ `exp_type`: should be 'ODMR' or 'NV' 
    `num` should be the experiment number.
    `date` should be a string such as '210609' for an experiment done on 9th June 2021.
    `raw`: default False, determine if the raw counts file should be returned, if it exists.
    `cache`: If True (default) then stores a copy of the found file on the local device
    at `DUMP_PATH`.
    `return_offline`: If True (default) then wait for the file copy to complete, 
        and return the offline address. If caching is disabled and no offline file is
        found, returns an error. If False, returns the network path or offline
        path, whichever is found, with preference for offline.
    """
    regex_search_str = r".*_(\d+)\D.*txt"
    # Try search the offline location.
    offline_name = get_offline_name(exp_type, num, date, raw, contains)
    offline_files = os.listdir(DUMP_PATH)
    if offline_name in offline_files:
        return DUMP_PATH + offline_name
    if return_offline and not cache:
        raise FileNotFoundError("Offline file requested without caching enabled, and no offline file could be found.")
    # If not, look for it in the network drive.
    network_loc = NETWORK_PATH + date + '/'
    try:
        network_files = os.listdir(network_loc)
        network_files = [f for f in network_files if exp_type in f]
        network_files = [f for f in network_files if contains in f]
        num_matches = [re.match(regex_search_str, f) for f in network_files]
        network_files = [m.group(0) for m in num_matches if (m != None and m.group(1) == str(num))]
        if len(network_files) >= 1:
            file = [f for f in network_files if (("raw" in f) == raw)]
        elif len(network_files) == 0:
            file = False
        else:
            file = network_files[0]
    except FileNotFoundError:
        date_exists = False
        if verbose:
            print("File not in network folder. ", end="")
        file = False
    else:
        date_exists = True

    parent_dir = network_loc
    if not file:
        # Try other paths
        for path in EXTRA_PATHS:
            p_path = path + date + "/"
            try:
                file_list = os.listdir(p_path)
            except FileNotFoundError:
                # date_exists = False
                continue
            else:
                date_exists = True

            
            file_list = [f for f in file_list if exp_type in f]
            file_list = [f for f in file_list if contains in f]
            num_matches = [re.match(regex_search_str, f) for f in file_list]

            file_list = [m.group(0) for m in num_matches if (m != None and m.group(1) == str(num))]
            # file_list = [f for f in file_list if re.match(".*_(\d+)[.,_].*", f).group(2)]
            if len(file_list) >= 1:
                file = [f for f in file_list if (("raw" in f) == raw)]
                parent_dir = p_path
                break
            elif len(file_list) == 0:
                file = False
                continue
            else:
                file = file_list[0]
                parent_dir = p_path
                break
    if not file:
        raise KeyDoesNotExistError("Could not find the file matching:\n"
            f"date: {date}\n"
            f"type: {exp_type}\n"
            f"num:  {num}\n"
            f"raw:  {raw}\n", date_exists=date_exists, n_exists=(False if date_exists else None))

    if type(file) == list:
        if len(file) > 1:
            raise AmbiguousSpecificationError(f"Multiple ({len(file)}) files were found matching:\n"
                f"date: {date}\n"
                f"type: {exp_type}\n"
                f"num:  {num}\n"
                f"raw:  {raw}\n")
        else:
            file = file[0]

    file = parent_dir + file
    
    if cache:
        fc = threading.Thread(target=copy_file, args=(file, DUMP_PATH+offline_name))
        fc.start()
        if return_offline:
            if verbose:
                print("Caching...")
            fc.join()
            if verbose:
                print("Done.")
            file = DUMP_PATH + offline_name
    return file



def copy_file(source, dest):
    shutil.copy(source, dest)
    
if __name__ == "__main__":
    find_file("NV", 16, "210609", True, cache=True)