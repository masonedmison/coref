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


def ment_in_set(ment, lits_set):
    """return True if a entity literal is contained in mention"""
    for lit in lits_set:
        if lit in ment:
            return True
    return False


def list_in_set(mentions_l, rel_terms):
    """Check that at least one relevant term (from termite) exists within a cluster and return list 
    if so, else return an empty list"""
    one = False
    for ment in mentions_l:
        ment_clean = ment.text.lower().strip()
        if ment_in_set(ment_clean, rel_terms):
            one = True
            break
    if one:
        return mentions_l
    else:
        return list()


def nc_to_conll(coref_clusts, termite_anns):
    """convert neuralcoref clusters to a tabular conll format sorted by mention index"""
    pass


def termite_to_conll(termite_res):
    """convert termite xml results to a tabluar conll file sorted by mention index"""
    pass
