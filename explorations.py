import spacy
import neuralcoref
from tests.abs_test import pub_ab

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, greedyness=0.5)


"""playground for nueral coref"""

toy_sent = 'My sister has a dog. She loves him'

doc = nlp(pub_ab)

print(f'[TOY SENTENCE] {pub_ab}')
print('-'*35)

print('[Coref Cluster things]')
print('all clusters in doc', doc._.coref_clusters)
# print(f'scores {doc._.coref_scores}')
# print(f'ALL mentions in first clusters {doc._.coref_clusters[1].mentions}')
# print(f'last mention in first cluster {doc._.coref_clusters[1].mentions[-1]}')
# print(f'?? {doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main}')

print('-'*35)



# a peak at spans and their indices
for s in doc._.coref_clusters:
    print(f'[CLUSTER] {s}')
    for ss in s.mentions:
        print(f'[MENTION] {ss}')
        print('start of span' ,ss.start)
        print('end of span', ss.end)
        print('-'*35)

# print('[Tokens things]')
# token = doc[-1]
# print(f'is token "{token}" in coref clusters {token._.in_coref}')
# print(f'clusters that contain THIS token {token._.coref_clusters}')
#
# print('-'*35)
#
# print('[Span things]')
# span = doc[-1:]
# print(f'is span "{span}" in at least one coref mention {span._.is_coref}')
# print(f'Span of the most representative mention in the cluster {span._.coref_cluster.main}')
# print(f'cluster of most representative mention {span._.coref_cluster.main._.coref_cluster}')
#
# print('-'*35)