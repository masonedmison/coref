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
        print('[TEXT]', ent.text, '[LABEL]', ent.label_)


def misc_inpect(doc_):
    """display tags, mention_type, dependencies"""
    print('\n[NOUN CHUNKS]') 
    for chunk in doc.noun_chunks:
        print('\n[CHUNK TEXT]', chunk.text, '\n[ROOT.TEXT]', chunk.root.text, '\n[ROOT.DEP]',chunk.root.dep_, '\n[ROOT.HEAD.TEXT]', chunk.root.head.text) 
    print('-'*35)

    print('\n[POS TAGS]')
    for token in doc:
        print('\n[TOKEN.TEXT]', token.text, '\n[TOKEN.LEMMA]', token.lemma_, '\n[TOKEN.POS]', token.pos_,'\n[TOKEN.TAG]', token.tag_, '\n[TOKEN.DEP]', token.dep_)
    print('-'*35)
                


ner_inspect(doc)
misc_inpect(doc)
    
