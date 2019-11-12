"""helper and utitlity functions"""
import glob
import numpy as np
from operator import itemgetter
import pathlib

EVAL_DATA_PATH = "eval_data/train"  # assumes you have data stored in this dir 

def get_random_batch(n=10):
    """get n number of random text files from BioNlP eval data"""
    all_txt = glob.glob(f'{EVAL_DATA_PATH}/*.txt')
    r_indices = np.random.choice(len(all_txt), n, replace=False)
    return itemgetter(*r_indices)(all_txt)


def get_str_from_file(file_):
    """get string from file
    Returns: 
        tuple(<filename_without_ext.>, <text of file>)
    """    
    f_r = open(file_,'r')
    full_path = pathlib.Path(file_)
    f_name = full_path.stem
    f_str = f_r.read() 
    f_r.close()
    return (f_name, f_str)
