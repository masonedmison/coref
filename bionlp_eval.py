"""A script that evaluates coference resolution on BioNLP 2011 training data
If any changes are made please run tests found in tests/test_eval.py"""
from collections import namedtuple
import re
import pathlib
import os

"""
regarding matches:
    (1) Exact match:
        - begin(detected mention)= begin(gold mention) & end(detected mention)= end(gold mention)
    (2) Partial match based on minimal and maximal boundaries of gold mentions:
        - begin(detected mention)>=begin(maximal boundary) & end(detected mention)<=end(maximal boundary)
        - begin(detected mention)<=begin(minimal boundary) & end(detected mention)>=end(minimal boundary)   
"""
# -----------------named tuples--------------------------------
cluster = namedtuple('Cluster', ('ant_span_', 'anaph_span_'))
span_ = namedtuple('Span', ('beg', 'end'))
# --------------------------------------------------------------

pos_neg_dict = dict(true_pos=0, false_pos=0, false_neg=0)
# -----------------functions specific to neural coref---------------
def commutative_pairing(cluster_list):
    """ Takes a list of clusters and pairs them off commutatively. This function expects a coref_clusters object (a list of clusters) that has been resolved by spacy
    e.g. [t1,t2,t3,t4] --> (t1,t2), (t1,t3), (t1,t4).
    Args:
        cluster_list: a list of cluster resolved by Spacy
        strip_spacy_attrs: if True, strip spacy objects to text word representation only
    Returns:
        list of tuples with len == 2 where members are SpaCY span objects. 
    """
    assert type(cluster_list) == list  # quick type check
    
    return [ (cluster_list[0], cluster_list[i]) for i in range(1, len(cluster_list)) ]


def word_to_char_indices(words_pair, container_text):
    """Extracts character index spans for a pair of mentions (spacy span objects)
    This function also drops the string value - keep this in mind for future development
    Args: 
        words_pair: word or phrase that to be convereted
        container_text: text where the word or phrase is contained in
    Returns:
        namedtuple cluster  
    """
    s_b_str = words_pair[0].text
    s_index_b = container_text.index(s_b_str)
    ant_span = span_(s_index_b, s_index_b+len(s_b_str))  # ant span

    s_e_str = words_pair[1].text
    s_index_e = container_text.index(s_e_str)
    anaph_span = span_(s_index_e, s_index_e+len(s_e_str))  # anaph span

    return cluster(ant_span, anaph_span)


def coref_clusters_to_spans(coref_clusts, container_text):
    """"Takes list of cluster objects and returns a list of cluster namedtuple objects where memebers are span_ objects
    Note that this function more or less iterates over the coref_cluster object and passses each cluster through commutative pairing and word_to_char indices
    Args:
        coref_clusts: a list of clusts resolved by neural coref
        container_text: abstract currently processed through nlp object
    Returns:
        set of cluster objects where members are span_ objects
    """
    char_span_clusters = set()
    for cl_d in coref_clusts:
        cp = commutative_pairing(cl_d.mentions)  
        for c in cp:
            char_span_cl = word_to_char_indices(c, container_text)
            char_span_clusters.add(char_span_cl)
    return char_span_clusters
# --------------------------------------------------------------------------


# -----------------functions specific to bionlp dataset-------------
def get_a2_file(a2_file):
    """ Takes full path to file of current abstract that is being processed and gets a2 file
    Args:
        f_path: Path to current file being processed
    Returns:
        str - Corresponding *.a2 
    """
    path = pathlib.Path(a2_file)
    f_name = path.stem
    a2_path = path.parent.joinpath(f'{f_name}.a2')
    return a2_path


def get_coref_spans(a2_file):
    """Extracts coreferring expressions from a2 file with mention indices
    Args:
        a2_file: path to a2_file 
    Returns:
        Set of clusters where members are cluster namedtuple objects
    """
    clusters = set()

    def get_term_spans(a2_reader):
        term_spans = dict()
        offset = 0 
        for l in a2_reader:
            span_obj = re.search(r'(T[0-9][0-9]?)\s*Exp\s*([0-9]{1,4}\s*[0-9]{1,4})', l)
            if span_obj:
                term = span_obj.group(1)
                span_str = span_obj.group(2)
                span_str_spli = span_str.split(' ')  # where begin span at i = 0 and end at i = 1 of span_str_split
                term_spans[term] = span_(int(span_str_spli[0]), int(span_str_spli[1]))
                offset += len(l)
            else:
                return term_spans, offset

    if os.stat(a2_file).st_size == 0: return clusters  # if file is empty return an empty set

    with open(a2_file, 'r') as a2:
        term_spans, offset = get_term_spans(a2)  # gets all term spans
        a2.seek(0)   # a dumb way to backtrack one line 
        a2.seek(offset)
        for line in a2:  # reader picks up where term spans ended
            if re.search(r'Coref Anaphora:', line):
                anaph_t_m = re.search(r'(Coref Anaphora:)(T[0-9][0-9]?[0-9]?)', line)
                anaph_t = anaph_t_m.group(2)
                antecedent_t_m = re.search(r'(Antecedent:)(T[0-9][0-9]?[0-9]?)', line)
                antecedent_t = antecedent_t_m.group(2)
                ant_span = term_spans[antecedent_t]
                anaph_span = term_spans[anaph_t]
                c = cluster(ant_span, anaph_span)
                clusters.add(c)
    return clusters


def atom_link_detector(pred_cluster, gold_clusters):
    """Checks if predicted cluster is an atom link
    Args:
        pred_cluster: a single predicted cluster
        gold_clusters: all gold clusters for a given doc
    Returns:
        True is atom link is found, otherwise False
    """
    pass


def within_min_span(pred_span, gold_span):
    """Detect if predicited mention span meets the 'minumum span' len declared in annotated data
    Args:
        pred_span: predicted span
        gold_span: mention span in annotated data
    Returns:
        True is span is within minimum span and False if is not
    """
    pass
# ------------------------------------------------------------------------

# ----------------true positive negative comparisons------------------------------------
""""both pred and gold clusters are a list of cluster objects where members are ant_span_ and anaph_span_ where span_ """
def get_true_pos(pred_clusters, gold_clusters):
    """Find all correct predictions
    Returns:
        Number of correct predicted coreference expressions
    """
    assert type(pred_clusters) and type(gold_clusters) == set  # both args should be sets

    return len(pred_clusters.intersection(gold_clusters))  # return length of members in both sets, ie true positives


# for every incorrect predicition = False Positive 
def get_false_pos(pred_clusters, gold_clusters):
    """Find incorrect predictions
    Returns:
        Number of incorrect predicted coreference expressions
    """
    return len(pred_clusters.difference(gold_clusters))


# for each prediction we did not get = False Negative 
def get_false_neg(pred_clusters, gold_clusters):
    """Find missed predictions
    Returns:
        Number of undetected coreference expressions 
    """
    return len(gold_clusters.difference(pred_clusters))
# ----------------------------------------------------------------------------------


# ----------------------accuracy metrics------------------------------
def precision(true_pos, false_pos):
    """Calculates Precision metric scocre
    Args: 
        true_pos: Correct predictions according Gold
        false_pos: predicted positives that are no in Gold
    Returns:
        precision metrics score
    """
    return true_pos/(true_pos+false_pos)


def recall(true_pos, false_neg):
    """computes recall metric score
    Args:
        true_pos: Correct predicitions according to Gold
        false_neg: missed predictions according to Gold
    Returns:
        recall metric score
    """
    return true_pos/(true_pos+false_neg)


def f1(prec, rec):
    """ Calculate F1 score or Harmonic Mean
    Args:
        prec: precision score
        rec: recall score
    Returns:
        f1 metric score
    """
    return 2 * (prec*rec)/(prec+rec)
# ------------------------------------------------------------------
