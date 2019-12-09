"""quick and dirty error analysis"""
from datetime import datetime
import pandas as pd
import spacy
import neuralcoref
from bionlp_eval import (coref_clusters_to_spans, get_a2_file, get_coref_spans, cluster_comparison, f1_, precision, recall)
from utils import get_random_batch, get_str_from_file

# modify model and codes used here
MODEL = 'ner_models/jnlpba_ner'
prune_by_ner = {'DNA', 'RNA', 'PROTEIN'}
####

####
nlp = spacy.load(MODEL)
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
####

f_batch = get_random_batch(n=20)

# xlsx workbook
writer = pd.ExcelWriter(f'results/error_analysis/EA_{datetime.now()}.xlsx', engine='xlsxwriter')


def process_txt_files(txt_files):
    """takes an iterable containing paths txt files"""
    for f in txt_files:
        print(f'[PROCESSING FILE] {f}')
        f_op = open(f, 'r', encoding='utf-8')
        f_str = f_op.read()
        f_op.close()
        doc = nlp(f_str)
        # see mentions
        pred_ents = [e.text for e in doc.ents]  
        ####
        # funky bug where files with 1 line throw a TypeError
        try:
            nc_clusts = coref_clusters_to_spans(doc._.coref_clusters, doc.text,
                    prune_by_ner=prune_by_ner) # neural coref
            a2_f = get_a2_file(f)  # get corresponding annotated file 
            gold_clusts, min_spans = get_coref_spans(a2_f)  # bionlp
        except TypeError as te:
            print(f'[FILE SKIPPED] {f}')
            continue  # muddle about - dont process anyting else in this loop
        # comparison
        pos_neg_dict = cluster_comparison(nc_clusts, gold_clusts, min_spans, debug=True)  # get positive negatives
        ####
        print(f'[FINISHED PROCESSING FILE] {f}')
        # get literal forms for series of clusters
        nc_lits = get_lit_forms(nc_clusts, f_str)
        gold_lits = get_lit_forms(gold_clusts, f_str)
        # put together dict
        d = dict(pred=list(nc_clusts), pred_lit=nc_lits, gold=list(gold_clusts), gold_lits=gold_lits)
        d.update(pos_neg_dict)
        d.update(pred_ents=pred_ents)
    # serialize res
        add_data(d, f)
    writer.save()


def add_data(d, f_name):
    df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))
    df.to_excel(writer, sheet_name=f_name.split('/')[-1])


def get_lit_forms(clusters, container_text):
    """takes list of clusters, returns list of literal form clusters"""
    lit_l = []
    for cl in clusters:
        # get ant lit
        ant_lit = container_text[cl.ant_span_.beg : cl.ant_span_.end]
        anaph_lit = container_text[cl.anaph_span_.beg : cl.anaph_span_.end] 
        lit_l.append((ant_lit, anaph_lit))
    return lit_l


if __name__ == '__main__':
    neuralcoref.add_to_pipe(nlp)
    process_txt_files(f_batch)
