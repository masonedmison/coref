"""A script that annotates an anaphoric expression to its predicted antecedent
If you would like to add this functionality as part of a pipline - simply pass txt into annotate_txt method
otherwise the script can be called from the command line - pass input txt as file or string as argument with --input flag"""

import argparse
import spacy
import neuralcoref
from termite import (get_rel_types)
from utils import ment_in_set, list_in_set


def load_model(txt):
    nlp = spacy.load('en_core_sci_md')
    neuralcoref.add_to_pipe(nlp)
    return nlp(txt)


def ants_anaph_end_char(doc, args): 
    global rel_terms
    clusts = doc._.coref_clusters
    if not bool(clusts):
        return []
    # get ant mappings to end of anaph exps
    ant_to_end = []
    for cl_d in clusts:
        cl_list = cl_d.mentions
        if args.termite_prune:
            cl_list = list_in_set(cl_d.mentions, rel_terms)
        if len(set(cl_list)) <= 1:
            continue
        ant = cl_list[0]
        for i in range(1, len(cl_list)):
            w = cl_list[i]
            w_clean = w.text.strip().lower()
            if args.prune_same_lit and w_clean == ant.text.strip().lower():
                continue
            # elif args.termite_prune and not ment_in_set(w_clean, rel_terms):
            #     continue  # kick out we dont care about this stinkin mention!
            ant_to_end.append((ant.text, w.end_char))
    ant_to_end = sorted(ant_to_end, key=lambda x: x[1], reverse=True)  # sort when breaks out of loop
    return ant_to_end


def annotate_txt(txt):
    # check if ant_to_anaph is empty
    global ant_to_end_c
    if not bool(ant_to_end_c):
        return txt

    for ant, end_i in ant_to_end_c:
        bef = txt[:end_i]
        aft = txt[end_i:].strip()
        txt = f"{bef}(({ant})) {aft}"
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Annotate coreffering expressions.')

    parser.add_argument('--input', help='Enter path to .txt file or raw text')
    parser.add_argument('--prune_same_lit', help='Prune same literal form or not - True or False', default=True)
    parser.add_argument('--termite_prune', 
    help='True if prune by termite NER is desired',
    default=False)
    parser.add_argument('--termite_rel_types', help='keep only relevant termite types, default values are "DRUG", "DRUGFIND", "INDICATION", "GENE", "CHEMREC", "GENEBOOST", "DRUGTYP", "PROTYP"',
    nargs='+', type=str, default=["DRUG", "DRUGFIND", "INDICATION", "GENE", "CHEMREC", "GENEBOOST", "DRUGTYP", "PROTYP"])
    
    args = parser.parse_args()
    input_ = args.input

    if input_.endswith('.txt'):
        with open(input_, 'r') as infile:
            input_ = infile.read()

    rel_terms = get_rel_types(input_, args.termite_rel_types) if args.termite_prune else set()

    print(rel_terms)
    print('ARGS CHECK', args.termite_rel_types)

#     input_ = """Leukotriene B4 stimulates c-fos and c-jun gene transcription and AP-1 binding activity in human monocytes. 
# We have examined the effect of leukotriene B4 (LTB4), a potent lipid proinflammatory mediator, on the expression of the proto-oncogenes c-jun and c-fos. In addition, we looked at the modulation of nuclear factors binding specifically to the AP-1 element after LTB4 stimulation. LTB4 increased the expression of the c-fos gene in a time- and concentration-dependent manner. The c-jun mRNA, which is constitutively expressed in human peripheral-blood monocytes at relatively high levels, was also slightly augmented by LTB4, although to a much lower extent than c-fos. The kinetics of expression of the two genes were also slightly different, with c-fos mRNA reaching a peak at 15 min after stimulation and c-jun at 30 min. Both messages rapidly declined thereafter. Stability of the c-fos and c-jun mRNA was not affected by LTB4, as assessed after actinomycin D treatment. Nuclear transcription studies in vitro showed that LTB4 increased the transcription of the c-fos gene 7-fold and the c-jun gene 1.4-fold. Resting monocytes contained nuclear factors binding to the AP-1 element, but stimulation of monocytes with LTB4 induced greater AP-1-binding activity of nuclear proteins. These results indicate that LTB4 may regulate the production of different cytokines by modulating the yield and/or the function of transcription factors such as AP-1-binding proto-oncogene products."""
# test path -- /home/medmison690/pyprojects/coref/eval_data/train/PMID-1618911.txt
# test path (WIN): C:\Users\EDMISML\PycharmProjects\coref\eval_data\train\PMID-7662982.txt
# test command line execution python coref_annotater.py --input C:\Users\EDMISML\PycharmProjects\coref\eval_data\train\PMID-7662982.txt --termite_prune True
    doc = load_model(input_)
    ant_to_end_c = ants_anaph_end_char(doc, args)
    print('ANT MAPPING', ant_to_end_c)
    annotated_txt = annotate_txt(doc.text)

    print(annotated_txt)
