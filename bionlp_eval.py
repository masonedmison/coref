"""A script that evaluates coference resolution on BioNLP 2011 training data"""

"""
regarding matches:
    (1) Exact match:
        - begin(detected mention)= begin(gold mention) & end(detected mention)= end(gold mention)
    (2) Partial match based on minimal and maximal boundaries of gold mentions:
        - begin(detected mention)>=begin(maximal boundary) & end(detected mention)<=end(maximal boundary)
        - begin(detected mention)<=begin(minimal boundary) & end(detected mention)>=end(minimal boundary)   
"""
# --------------------------------------------------
# PSUEDO CODE 

# Assuming out of of doc._coref_clusters...

def commutative_pairing(cluster_list, strip_spacy_attrs=True):
    """ Takes a list of clusters and pairs them off commutatively.
    e.g. [t1,t2,t3,t4] --> (t1,t2), (t1,t3), (t1,t4).
    This function expects a coref_clusters object (a list of clusters) that has been resolved by spacy
    Args:
        cluster_list: a list of cluster resolved by Spacy
        strip_spacy_attrs: if True, strip spacy objects to text word representation only
    Returns:
        list of tuples with len == 2 where members are SpaCY span objects or text depending on strip_spacy_attrs. 
    """
    pass


# convert word index to char index - SANITY CHECK
def word_to_char_indices(words_pair, container_text):
    """Extracts character index spans for a pair of words
    Current implementation only accepts type(word[0]) == str. 
    This function also drops the string value - keep this in mind for future development
    Args: 
        words_pair: word or phrase that to be convereted
        container_text: text where the word or phrase is contained in
    Returns:
        tuple with each word as a character index span
    """
    pass

# get corresponding a2 file 
def get_true_mentions(f_path):
    """ Takes full path to file of current abstract that is being processed and gets a2 file
    Args:
        f_path: Path to current file being processed
    Returns:
        str - Corresponding *.a2 
    """
    pass
    
# slice relevant section from line containting beginning with R1 to end of doc
def get_gold_coref_exps(a2_f_path):
    """extracts relevant area of .a2 txt document containing coreferring expressions
    Args:
        a2_f_path: file path to a2 file 
    Returns:
        section of document containing coref expressions
    """
    pass

# extract spans of expressions for each coref exp
def get_coref_spans(coref_exp):
    """Get pair clusters containing character indices from a coref_exp (BART format)
    Args:
        coref_exp: a coreference expression, e.g. 'R3	Coref Anaphora:T19 Antecedent:T15'
    Returns:
        tuple of len == 2 where each entry is a character index span
    """
    pass
# where AD = Annotated Dataset and NCP= Neural Coref Predicitions

# AD = True ---- NCP= Predicted
# when comparing predicited vs true 
# ----> have a function to detect atom links, ie Surface ->(T3,T2), (T2, T1) == Atom-> (T3,T1)

# ----------------------------------------------------
# get stuff to calculate metrics (prec, recall, f1)

# for every prediction that is right = True Positive

# for every incorrect predicition = False Positive 

# for each prediction we did not get = False Negative 

# ----------------------------------------------------

# ----------------------------------------------------
# metrics
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
# ----------------------------------------------------
