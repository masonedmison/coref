"""Script that runs neural coref coreference resolution library on bionlp dataset"""
import glob
import logging
import os
import spacy
import neuralcoref
from bionlp_eval import (coref_clusters_to_spans, get_a2_file, get_coref_spans, cluster_comparison, f1_, precision, recall)

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


def get_txt_files(path_to_files):
    """returns iterator which yeilds txt files in given dir"""
    return glob.iglob(os.path.join(path_to_files, '*.txt'))


def calculate_metrics(pos_neg_dict):
    """calculates metrics and writes to txt file"""
    tp = pos_neg_dict['true_pos']
    fp = pos_neg_dict['false_pos']
    fn = pos_neg_dict['false_neg']

    prec = precision(tp, fp)
    rec = recall(tp, fn)

    f1 = f1_(prec, rec)

    print('F1', f1)
    print('\nPRECISION', prec)
    print('\nRECALL', rec)


def process_txt_files(txt_files):
    """takes an iterable containing paths txt files"""
    for f in txt_files: 
        logging.info(f'[PROCESSING FILE] {f}')
        f_op = open(f, 'r', encoding='utf-8')
        f_str = f_op.read()
        f_op.close()
        doc = nlp(f_str)
        # funky bug where files with 1 line throw a TypeError
        try:
            # neural coref 
            nc_clusts = coref_clusters_to_spans(doc._.coref_clusters, doc.text)
            # bionlp
            a2_f = get_a2_file(f)
            gold_clusts, min_spans = get_coref_spans(a2_f)
        except TypeError as te:
            logging.warn(te)
            logging.warn(f'[FILE SKIPPED] {f}')
            print(f'[FILE SKIPPED] {f}')
            pass
        # comparison
        pos_neg_dict = cluster_comparison(nc_clusts, gold_clusts, min_spans)
        logging.info(f'[FINISHED PROCESSING FILE] {f}')
    calculate_metrics(pos_neg_dict)


def main():
    txt_it = get_txt_files('eval_data/train')
    process_txt_files(txt_it)


if __name__ == '__main__':
    logging.basicConfig(
     filename=f'log/bionlp_eval.log',
     level=logging.INFO, 
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )
    logger = logging.getLogger('bionlp_eval')
    logger.setLevel(logging.DEBUG)
    
    main()
