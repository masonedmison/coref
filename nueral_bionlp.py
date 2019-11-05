"""Script that runs neural coref coreference resolution library on bionlp dataset"""
import glob
import os
import spacy
import neuralcoref
from bionlp_eval import coref_clusters_to_spans, get_a2_file, get_coref_spans

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def get_txt_files(path_to_files):
    """returns iterator which yeilds txt files in given dir"""
    return glob.iglob(os.path.join(path_to_files, '*.txt'))


def process_txt_files(txt_files):
    """takes an iterable containing paths txt files"""
    for f in txt_files: 
        f_op = open(f, 'r', encoding='utf-8')
        f_str = f_op.read()
        f_op.close()
        doc = nlp(f_str)    
        # neural coref 
        nc_clusts = coref_clusters_to_spans(doc._.coref_clusters, doc.text)
        # bionlp
        a2_f = get_a2_file(f)
        gold_clusts, min_spans = get_coref_spans(a2_f)  


def main():
    txt_it = get_txt_files('eval_data/train')
    process_txt_files(txt_it)


if __name__ == '__main__':
    main()