import spacy
import neuralcoref
from tests.abs_test import pub_ab

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, greedyness=0.5)


"""playground for nueral coref"""

toy_sent = 'My sister has a furry-dog. She loves him. Her favorite color is blue. She likes this because it is pretty. The dog is pretty good overall.'

doc = nlp(toy_sent)

print(doc._.coref_clusters)


# a peak at spans and their indices
for i,s in enumerate(doc._.coref_clusters):
    # print(f'[CLUSTER] {s}')
    print(f'antecedent {doc._.coref_clusters[i].mentions[0]}')
    for i,ss in enumerate(s.mentions):
        if i == 0:
            continue
        print(f'[Anaphora] {ss}')
        print('start of span' ,ss.start)
        print('end of span', ss.end)
        print('-'*35)


word_ = 'furry-dog'
# get char i of furry-dog at (using str.index) -- span to len(furry-dog)
d_index = toy_sent.index(word_)
d_char_span = toy_sent[d_index:d_index+len(word_)]
print(d_char_span)