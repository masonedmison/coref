"""quick and dirty error analysis"""
from datetime import datetime
import pandas as pd
import spacy
import neuralcoref
from bionlp_eval import (coref_clusters_to_spans, get_a2_file, get_coref_spans, cluster_comparison, f1_, precision, recall)
from utils import get_random_batch, get_str_from_file

# modify model and codes used here
MODEL = 'en_core_sci_md'
CODES = 'alg_plfp_glfpma'  # seperated by '_' -- see results/res_codes.txt
####

####
# change model as needed
nlp = spacy.load(MODEL)
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
####
neuralcoref.add_to_pipe(nlp)

f_batch = get_random_batch(n=20)

# xlsx workbook
writer = pd.ExcelWriter(f'results/error_analysis/EA_{datetime.now()}.xlsx', engine='xlsxwriter')


def process_txt_files(txt_files):
    """takes an iterable containing paths txt files"""
    # total_pred = 0
    # total_gold = 0
    for f in txt_files:
        print(f'[PROCESSING FILE] {f}')
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
            print(f'[FILE SKIPPED] {f}')
            continue  # muddle about - dont process anyting else in this loop
        # comparison
        pos_neg_dict = cluster_comparison(nc_clusts, gold_clusts, min_spans, debug=True)  # get positive negatives

        # keep track of total
        # total_pred += len(nc_clusts)
        # total_gold += len(gold_clusts)
        ####
        print(f'[FINISHED PROCESSING FILE] {f}')
        # put together dict
        d = dict(pred=list(nc_clusts), gold= list(gold_clusts))
        d.update(pos_neg_dict)
    # serialize res
        add_data(d, f)
    writer.save()


def add_data(d, f_name):
    df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))
    df.to_excel(writer, sheet_name=f_name.split('/')[-1])


if __name__ == '__main__':
    process_txt_files(f_batch)