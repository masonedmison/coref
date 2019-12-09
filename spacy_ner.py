import spacy
import neuralcoref
from utils import get_random_batch
from spacy.language import Language
from spacy.tokenizer import Tokenizer

# ner_mod = spacy.load('en_ner_jnlpba_md')
# ner_pipe = ner_mod.pipeline[0][1]
# nlp = spacy.load('en_core_sci_md')
# nlp.remove_pipe('ner')
# nlp.add_pipe(ner_pipe, name="ner", last=True)

# nlp.to_disk('ner_models/jnlpba_ner')

nlp = spacy.load('ner_models/craft_ner')



batch = get_random_batch(n=5)

# nlp = spacy.load('en_ner_craft_md')
neuralcoref.add_to_pipe(nlp)



f_re = open(batch[0], 'r')
f_str = f_re.read() 

print(batch[0])
# text = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. "

doc = nlp(f_str)


def display_ents(doc_):
    print('ENTS')
    for ent in doc_.ents:
        print(ent.label_)


def display_coref(doc_):
    print('CLUSTS')
    cl_l = doc_._.coref_clusters
    for cl_d in cl_l:
        print(cl_d)


display_ents(doc)
# display_coref(doc)
