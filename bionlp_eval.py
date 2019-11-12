"""A script that evaluates coference resolution on BioNLP 2011 training data
If any changes are made please run tests found in tests/test_eval.py
regarding matches:
    (1) Exact match:
        - begin(detected mention)= begin(gold mention) & end(detected mention)= end(gold mention)
    (2) Partial match based on minimal and maximal boundaries of gold mentions:
        - begin(detected mention)>=begin(maximal boundary) & end(detected mention)<=end(maximal boundary)
        - begin(detected mention)<=begin(minimal boundary) & end(detected mention)>=end(minimal boundary)
"""
from collections import namedtuple
import re
import pathlib
import os


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
        cluster_list: a list of cluster resolved by neural coref
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
        cp = prune_same_form(cp)  # prune list of all mentions that are of same form
        for c in cp:
            char_span_cl = word_to_char_indices(c, container_text)
            char_span_clusters.add(char_span_cl)
    return char_span_clusters
# --------------------------------------------------------------------------


# -----------------functions specific to bionlp dataset-------------
def get_a2_file(curr_file):
    """ Takes full path to file of current abstract that is being processed and gets a2 file
    Args:
        f_path: Path to current file being processed
    Returns:
        str - Corresponding *.a2
    """
    path = pathlib.Path(curr_file)
    f_name = path.stem
    a2_path = path.parent.joinpath(f'{f_name.strip()}.a2')
    return a2_path


def get_coref_spans(a2_file):
    """Extracts coreferring expressions from a2 file with mention indices
    Args:
        a2_file: path to a2_file
    Returns:
        clusters: Set of clusters where members are cluster namedtuple objects and min_spans: min_spans look up
    """
    clusters = set()
    min_spans = dict()  # where span_ object is mapped to it's minumum span (if any)

    def get_term_spans(a2_reader):
        term_spans = dict()
        offset = 0
        for l in a2_reader:
            l_spli = l.split('\t')
            if 'T' in l_spli[0]:
                ws_str = l_spli[1].replace('Exp', '').strip()
                ws_spli = ws_str.split(' ')
                ws = span_(int(ws_spli[0]), int(ws_spli[1]))
                term_spans[l_spli[0]] = ws
                #### 
                # handle min span if any
                if len(l_spli) == 5:
                    ms_str = l_spli[3]
                    ms_spli = ms_str.split(' ')
                    min_spans[ws] = span_(int(ms_spli[0]), int(ms_spli[1]))
                else:
                    min_spans[ws] = None
                ####
                offset += len(l)
            else:
                return term_spans, offset

    if os.stat(a2_file).st_size == 0: return clusters, min_spans  # if file is empty return empty clusters and min_spans

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
    return clusters, min_spans


def atom_link_detector(pred_cluster, gold_clusters):
    """Checks if predicted cluster is an atom link
    A cluster is an atom link when (c1,c3) when defined coref expression exist in gold --> (c1,c2), (c2,c3)
    Args:
        pred_cluster: a single predicted cluster
        gold_clusters: gold clusters for a given doc as iterable
    Returns:
        True if atom link is found, otherwise False
    """
    g = None
    for cl in gold_clusters:
        if cl.ant_span_ == pred_cluster.ant_span_:
            g = cl
            break

    if g is None:
        return False  # if still None then no matching ant span was found

    # search for links
    l = g
    for c in gold_clusters:
        if l.anaph_span_ == c.ant_span_:
            if pred_cluster == cluster(g.ant_span_, c.anaph_span_):
                return True
            else:
                l = c
    return False


def within_min_span(pred_span, gold_span, min_spans):
    """Detect if predicited mention span meets the 'minumum span' len declared in annotated data
    Args:
        pred_span: predicted span
        gold_span: gold span in annotated data
        min_spans: dictionary that maps each whole span to it's min span. None val if no min span for mention
    Returns:
        True is span is within minimum span and False if not
    """

    m_span = min_spans[gold_span]
    if m_span is None:
        m_span = gold_span
    # rules as defined in top level doc string
    if pred_span.beg >= gold_span.beg and pred_span.end <= gold_span.end:
        if pred_span.beg <= m_span.beg and pred_span.end >= m_span.end:
            return True

    return False
# ------------------------------------------------------------------------


# ---------------------general functions----------------------------------
def prune_same_form(clusts):
    """prune clusters that have the same literal form
    Args:
        clust: list of Spacy mentions OR a tuple - types checked before processing if conducted
    Return:
        pruned clusts
        ...
        """
    if type(clusts) is list:  #  if list, incoming from predicted (neural coref) 
        prune_copy = clusts.copy()
        for cl in clusts:
            if cl[0].text.lower() == cl[1].text.lower():
                prune_copy.remove(cl)
        return prune_copy
    elif type(clusts) is tuple:  # if clusts of type tuple - it is incoming from gold
        if clusts[0].text.lower() == clusts[1].text.lower():
            return None
        else:
            return clusts
# ------------------------------------------------------------------------


# ----------------------accuracy metrics------------------------------
def precision(true_pos, false_pos):
    """Calculates Precision metric scocre
    Args:
        true_pos: Correct predictions according Gold
        false_pos: predicted positives that are no in Gold
    Returns:
        precision metrics score
    """
    if true_pos + false_pos == 0:
        return 0
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


def f1_(prec, rec):
    """ Calculate F1 score or Harmonic Mean
    Args:
        prec: precision score
        rec: recall score
    Returns:
        f1_ metric score
    """
    return 2 * ((prec*rec)/(prec+rec))
# ------------------------------------------------------------------


# ---------------------comparison drivers---------------------------
def min_span_bundle(pred_clust, gold_clusters, min_spans):
    """takes a single predicted cluster and checks if both clusters may exist within the minumum spans of all of the gold clusters"""
    for g_clust in gold_clusters:
        if within_min_span(pred_clust.ant_span_, g_clust.ant_span_, min_spans) and within_min_span(pred_clust.anaph_span_, g_clust.anaph_span_, min_spans):
            return g_clust
    return None


def cluster_comparison(pred_clusters, gold_clusters, min_spans, debug=False):
    """Compare predicted or detected mentions against annotated clusters in dataset for a single document
    Args:
        pred_clusters: clusters detected or resolved by neural coref
        gold_clusters: clusters annotated in bionlp dataset
    Returns:
        Number of true positives, false positives, false negatives
    """
    global pos_neg_dict
    # for debugging purposes, set all dict values back to zero
    if debug:
        for k in pos_neg_dict:
            pos_neg_dict.update({k:0})
    ####
    pred_cl_copy = set(pred_clusters)
    for pred_clust in pred_clusters:
        if pred_clust in gold_clusters:
            pos_neg_dict['true_pos'] += 1
            continue
        # minimum span check
        min_span_ex = min_span_bundle(pred_clust, gold_clusters, min_spans)
        if min_span_ex is not None:
            pos_neg_dict['true_pos'] += 1
            pred_cl_copy.remove(pred_clust)  # remove partial match and add gold clust matched on - we do this so we can do set difference for false positives
            pred_cl_copy.add(min_span_ex)
            continue  # kick out so we don't run atom_link_detection
        # atom link detection
        has_atom_link = atom_link_detector(pred_clust, gold_clusters)
        if has_atom_link:
            pos_neg_dict['true_pos'] += 1
            pos_neg_dict['false_neg'] -= 1  # blindly subtract one since atom link will not be considered in difference of gold and pred sets
        else:  # no exact or partial match so false positive
            pos_neg_dict['false_pos'] += 1

    false_negs = len(gold_clusters.difference(pred_cl_copy))
    pos_neg_dict['false_neg'] += false_negs


    return pos_neg_dict
# ------------------------------------------------------------------
