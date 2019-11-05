"""tests for the bionlp_eval.py scripts"""
from functools import partial
import pytest
import spacy
import neuralcoref
from bionlp_eval import (commutative_pairing, word_to_char_indices, cluster, span_, get_coref_spans,
coref_clusters_to_spans, atom_link_detector, within_min_span)


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
    a2_spans1, _ = get_coref_spans('eval_data/train/PMID-1315834.a2')

    assert a2_spans1 == {cluster(span_(560, 595), span_(597, 602)), cluster(span_(645, 679), span_(680, 685))}

    # an empty a2 file
    a2_spans2 = get_coref_spans('eval_data/train/PMID-2105946.a2')
    assert a2_spans2 == set() # should return an empty set


def test_min_spans():
    print('\n------testing min spans within get_coref_spans()--------')
    _, min_spans = get_coref_spans('eval_data/train/PMID-1372388.a2')

    ####
    # sample whole spans
    ws1 = span_(101,137)  # T20
    ws2 = span_(232, 286)  # T22
    ws3 = span_(582, 601)   # T24 
    ws4 = span_(747, 786)   # T26
    ws5 = span_(694, 697)
    ####
    # min spans
    ms1 = span_(131, 137)
    ms2 = span_(259, 286)
    ms3 = span_(592, 601)
    ms4 = span_(772, 786)
    ####

    assert min_spans[ws1] == ms1
    assert min_spans[ws2] == ms2
    assert min_spans[ws3] == ms3
    assert min_spans[ws4] == ms4

    assert min_spans[ws5] is None  # test for term with no min span


def test_atom_link():
    print('\n---------testing atom link detection-----------')
    # surface links
    sl1 = cluster(span_(224,226), span_(334,336))
    sl2 = cluster(span_(334, 336), span_(444,446))
    sl3 = cluster(span_(444,446), span_(554, 556))
    sl4 = cluster(span_(554, 556), span_(664, 666))
    sl5 = cluster(span_(333, 335), span_(553, 555))
    sl_clusters1 = [sl1, sl2, sl3, sl4, sl5]
    # atom links
    al1 = cluster(span_(224,226), span_(444,446))
    al2 = cluster(span_(224,226), span_(664,666))
    # no atom link
    nal1 = cluster(span_(333,335), span_(773, 775))
    nal2 = cluster(span_(111,222), span_(888, 999))

    assert atom_link_detector(al1, sl_clusters1) is True

    assert atom_link_detector(al2, sl_clusters1) is True

    assert atom_link_detector(nal1, sl_clusters1) is False

    assert atom_link_detector(nal2, sl_clusters1) is False


def test_min_span_detecting():
    print('\n------------testing min span detection-----------------')
    ####
    # span objects
    g_span = span_(100, 150)
    g_no_min = span_(400, 402)  # ensure no min is found, should always return False
    m_span = span_(120, 135)
    min_spans = {g_span: m_span, g_no_min:None}  # whole span maps to min span
    # predicted spans
    p_span1 = span_(115, 145)  # True
    p_span2 = span_(120, 135)  # True
    p_span3 = span_(122, 135)  # False
    p_span4 = span_(120, 134)  # False
    p_span5 = span_(110, 150)  # True
    ####
    
    wms = partial(within_min_span, gold_span=g_span, min_spans=min_spans)

    assert wms(p_span1) is True
    assert wms(p_span2) is True
    assert wms(p_span3) is False
    assert wms(p_span4) is False
    assert wms(p_span5) is True

    assert within_min_span(p_span1, g_no_min, min_spans) is False
    



