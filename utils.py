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


def prune_by_ent(span_list, ents):
    """takes a list of tuples where members are SpaCy span objects and prunes
    tuples where neither member is one of the ents passed in
    'DNA', 'CELL_TYPE', 'CELL_LINE', 'RNA', 'PROTEIN'
    Returns:
        copy of modified span_list
    """
    if not bool(span_list):
        return span_list

    span_l_copy = span_list.copy()

    # get ents for doc
    ent_lits = span_list[0][0].doc.ents
    # prune by passed in ents
    ent_lits = [e for e in ent_lits if e.label_ in ents]

    for tup in span_list:
        ent_found = False
        for ent in ent_lits:
            ent_clean = ent.text.lower().strip()
            if ent_clean in tup[0].text.lower().strip() or ent_clean in tup[1].text.lower().strip():
                ent_found = True
        if not ent_found:
            print('[tuple pruned]', tup)
            span_l_copy.remove(tup)
    return span_l_copy
