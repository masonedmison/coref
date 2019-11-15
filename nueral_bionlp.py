"""Script that runs neural coref coreference resolution library on bionlp dataset"""
import glob
import logging
import os
import spacy
import neuralcoref
from bionlp_eval import (coref_clusters_to_spans, get_a2_file, get_coref_spans, cluster_comparison, f1_, precision, recall)

# modify model and codes used here
MODEL = 'en_core_sci_lg'
CODES = 'alg_plfp_glfpma'  # seperated by '_' -- see results/res_codes.txt
####

####
# change model as needed
nlp = spacy.load(MODEL)
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
####
neuralcoref.add_to_pipe(nlp)


def get_txt_files(path_to_files):
    """returns iterator which yeilds txt files in given dir"""
    return glob.iglob(os.path.join(path_to_files, '*.txt'))


def calculate_metrics(pos_neg_dict, pred_gold_totals, write_res=False):
    """calculates metrics and writes to txt file"""
    tp = pos_neg_dict['true_pos']
    fp = pos_neg_dict['false_pos']
    fn = pos_neg_dict['false_neg']

    prec = precision(tp, fp)
    rec = recall(tp, fn)

    f1 = f1_(prec, rec)

    if write_res:
        write_results(f1, prec, rec, pred_gold_totals, pos_neg_dict)


def write_results(f1, precision, recall, pred_gold_totals, pos_neg_dict):
    with open(f'{MODEL}_{CODES}.txt', 'w') as out:
        out.write('[ACCURACY METRICS]')
        out.write(f'\n[F1] {f1*100}%')
        out.write(f'\n[PRECISION] {precision*100}%')
        out.write(f'\n[RECALL] {recall*100}%')
        # total clusters generated
        out.write(f'\n[TOTAL PREDICTED CLUSTERS] {pred_gold_totals[0]}')
        out.write(f'\n[TOTAL GOLD CLUSTERS] {pred_gold_totals[1]}')
        out.write(f'\n[TRUE POSITIVES] {pos_neg_dict["true_pos"]}') 
        out.write(f'\n[FALSE POSITIVES] {pos_neg_dict["false_pos"]}') 
        out.write(f'\n[FALSE NEGATIVES] {pos_neg_dict["false_neg"]}')


def process_txt_files(txt_files):
    """takes an iterable containing paths txt files"""
    total_pred = 0
    total_gold = 0
    for f in txt_files:
        logging.info(f'[PROCESSING FILE] {f}')
        f_op = open(f, 'r', encoding='utf-8')
        f_str = f_op.read()
        f_op.close()
        doc = nlp(f_str)
        # funky bug where files with 1 line throw a TypeError
        try:
            nc_clusts = coref_clusters_to_spans(doc._.coref_clusters, doc.text) # neural coref
            a2_f = get_a2_file(f)  # get corresponding annotated file 
            gold_clusts, min_spans = get_coref_spans(a2_f)  # bionlp
        except TypeError as te:
            logging.warn(te)
            logging.warn(f'[FILE SKIPPED] {f}')
            print(f'[FILE SKIPPED] {f}')
            continue  # muddle about - dont process anyting else in this loop
        # comparison
        pos_neg_dict = cluster_comparison(nc_clusts, gold_clusts, min_spans)  # get positive negatives
        # keep track of total
        total_pred += len(nc_clusts)
        total_gold += len(gold_clusts)
        ####
        logging.info(f'[FINISHED PROCESSING FILE] {f}')
    pred_gold_totals = (total_pred, total_gold)
    calculate_metrics(pos_neg_dict, pred_gold_totals, write_res=True)


def main():
    txt_it = get_txt_files('eval_data/train')  # iterator over all txt files in dir
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
