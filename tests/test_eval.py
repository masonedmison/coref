"""tests for the bionlp_eval.py scripts"""
import pytest
import spacy
import neuralcoref
from bionlp_eval import (commutative_pairing, word_to_char_indices, cluster, span_, get_coref_spans,
coref_clusters_to_spans)


@pytest.fixture(scope='session')
def set_up():
    """load in md spacy model. add neuralcoref to pipe"""
    print('\n--------------set up-----------------')
    print('\nloading spacy model - this may take a minute or so')
    nlp = spacy.load('en_core_web_md')
    neuralcoref.add_to_pipe(nlp, greedyness=0.5)
    print('\n-------------finished set up----------')
    return nlp


@pytest.fixture(scope='module')
def neurcoref_226(set_up):
    """return doc object of pubmed abstract 1313226"""
    f = open('/home/medmison690/pyprojects/coref/eval_data/train/PMID-1313226.txt')
    f_str = f.read()  # read the thing into as str
    doc = set_up(f_str)
    f.close()
    return doc  # return spacy pipeline with neural coref

def _get_cluster_mentions(coref_clusters):
    """helper to extract list of clusters from neuralcoref"""
    return [c.mentions for c in coref_clusters]

def test_cummutative_pairing1(neurcoref_226):  # bionlp_eval.commutative_pairing
    print('---------testing commutative pairing---------------')
    nc_clusters = neurcoref_226._.coref_clusters

    clusters_lists = _get_cluster_mentions(nc_clusters)
    
    cp1 = commutative_pairing(clusters_lists[0])
    assert cp1 == [ (clusters_lists[0][0], clusters_lists[0][1]) ]

    cp2 = commutative_pairing(clusters_lists[1])
    assert cp2 == [ (clusters_lists[1][0], clusters_lists[1][1]), (clusters_lists[1][0], clusters_lists[1][2]), (clusters_lists[1][0], clusters_lists[1][3]), (clusters_lists[1][0], clusters_lists[1][4]),
    (clusters_lists[1][0], clusters_lists[1][5]) ] 


def test_word_char_indices(neurcoref_226): # bionlp_eval.word_to_char_indices
    print('\n----------testing word to char indices---------------')
    nc_clusters = neurcoref_226._.coref_clusters
    clusters_lists = _get_cluster_mentions(nc_clusters)

    cp1 = commutative_pairing(clusters_lists[0])

    # ensure that text in clusters == character indexing
    # c is a cluster(ant_span, anaph_span) object where span('beg', 'end')
    for cp in cp1:
        cs = word_to_char_indices(cp, neurcoref_226.text)  # returns cluster object with ant and beg span
    
        ant_ = cs.ant_span_
        anaph_ = cs.anaph_span_
        ####
        # check return types 
        assert type(cs) == cluster
        assert type(ant_) == span_
        assert type(anaph_) == span_
        assert type(ant_.beg) and type(ant_.end) == int
        ####
        ####
        # check that char indices match up with output of spacy 
        assert neurcoref_226.text[ant_.beg: ant_.end] == cp[0].text  # once for ant (in commutative pair tuple)
        assert neurcoref_226.text[anaph_.beg: anaph_.end] == cp[1].text  # once for ant (in commutative pair tuple)
        ####

def test_get_coref_spans():
    print('\n-----------------testing get coref spans----------------------')
    a2_spans1 = get_coref_spans('eval_data/train/PMID-1315834.a2')

    assert a2_spans1 == {cluster(span_(560, 595), span_(597, 602)), cluster(span_(645, 679), span_(680, 685))}

    # an empty a2 file
    a2_spans2 = get_coref_spans('eval_data/train/PMID-2105946.a2')
    assert a2_spans2 == set() # should return an empty set

