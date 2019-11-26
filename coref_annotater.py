"""A script that annotates an anaphoric expression to its predicted antecedent"""
from collections import OrderedDict
import argparse
import spacy
import neuralcoref


def load_model(txt):
    nlp = spacy.load('en_core_sci_md')
    neuralcoref.add_to_pipe(nlp)
    return nlp(txt)


def ants_anaph_end_char(doc):
    clusts = doc._.coref_clusters
    # get ant mappings to end of anaph exps
    ant_to_end = {}
    for cl_d in clusts:
        cl_list = cl_d.mentions
        ant = cl_list[0] 
        ant_to_end[ant] = []
        for i in range(1, len(clusts)):
            w = cl_list[i]
            ant_to_end[ant].append(w.end_char) 
        ant_to_end[ant].sort()  # sort when breaks out of loop
    return ant_to_end


def get_max_i():
    global ant_to_end_c
    if not bool(ant_to_end_c):
        return None, None
    ant_to_end_c = sorted(ant_to_end_c.items(), key=lambda x: x[1][-1], reverse=True)
    ant_to_end_c = OrderedDict(ant_to_end_c) 
    f_key = list(ant_to_end_c.keys())[0]
    ret_val = ant_to_end_c[f_key][-1] 
    ant_to_end_c[f_key].remove(ret_val)
    if not bool(ant_to_end_c[f_key]):
        del ant_to_end_c[f_key]
    return f_key, ret_val


def annotate_txt(txt):
    # check if ant_to_anpah is empty
    global ant_to_end_c
    if not bool(ant_to_end_c):
        return txt

    ant, to_add = get_max_i()
    while to_add is not None:
        bef = txt[:to_add]
        aft = txt[to_add:]
        txt = f"{bef}({ant}) {aft}"
        ant, to_add = get_max_i()
    return txt


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Annotate coreffering expressions.')
    # parser.add_argument('--input', help='Enter path to .txt file or raw text')

    # args = parser.parse_args()
    # input_ = args.input

    # if input_.endswith('.txt'):
    #     with open(input_, 'r') as infile:
    #         input_ = infile.read()

    input_ = """Leukotriene B4 stimulates c-fos and c-jun gene transcription and AP-1 binding activity in human monocytes. 
We have examined the effect of leukotriene B4 (LTB4), a potent lipid proinflammatory mediator, on the expression of the proto-oncogenes c-jun and c-fos. In addition, we looked at the modulation of nuclear factors binding specifically to the AP-1 element after LTB4 stimulation. LTB4 increased the expression of the c-fos gene in a time- and concentration-dependent manner. The c-jun mRNA, which is constitutively expressed in human peripheral-blood monocytes at relatively high levels, was also slightly augmented by LTB4, although to a much lower extent than c-fos. The kinetics of expression of the two genes were also slightly different, with c-fos mRNA reaching a peak at 15 min after stimulation and c-jun at 30 min. Both messages rapidly declined thereafter. Stability of the c-fos and c-jun mRNA was not affected by LTB4, as assessed after actinomycin D treatment. Nuclear transcription studies in vitro showed that LTB4 increased the transcription of the c-fos gene 7-fold and the c-jun gene 1.4-fold. Resting monocytes contained nuclear factors binding to the AP-1 element, but stimulation of monocytes with LTB4 induced greater AP-1-binding activity of nuclear proteins. These results indicate that LTB4 may regulate the production of different cytokines by modulating the yield and/or the function of transcription factors such as AP-1-binding proto-oncogene products."""

    doc = load_model(input_)
    ant_to_end_c = ants_anaph_end_char(doc)
    annotated_txt = annotate_txt(doc.text)
    print(annotated_txt)
