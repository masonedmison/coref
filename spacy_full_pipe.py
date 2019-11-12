"""explorations of full SpaCY in neural coref pipeline"""
import neuralcoref
import scispacy
import spacy
from utils import (get_random_batch, get_str_from_file)



# instantiate spacy and add neuralcoref to pipeline
nlp = spacy.load('en_core_sci_md')
neuralcoref.add_to_pipe(nlp)

batch = get_random_batch(n=20)  # get n random pubmed abstracts

# single doc 
f_tup = get_str_from_file(batch[0])
doc = nlp(f_tup[1])

def ner_inspect(doc_):
    """display stuff about entities detected by SpaCy NER module"""
    for ent in doc_.ents:
        print('[TEXT]', ent.text,'[LABEL]', ent.label_)

ner_inspect(doc)
    