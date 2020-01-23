"""various methods to convert neural coref output and termite output into a data tuple that is compatible with 
Sanaphor (specifically within parse_corefs file)"""
import re


def _id_clean_format(main_mention, add_idx=True):
    pattern = r'[^a-zA-Z0-9]' #only extract alpha numeric characters
    if add_idx:
        st_end_concat = '_' + str(main_mention.start) + str(main_mention.end)
    else:
        st_end_concat = ""

    return re.sub(pattern, r' ', main_mention.text).lower() + st_end_concat


def _extract_nc_feats(nc_clusts):
    """takes neuralcoref coref_clusters objects, extracts relevant features
    Returns:
        doc_features a list of feature lists according to conll indexing"""
    doc_features = []
    for clust in nc_clusts:
        coref_clean = _id_clean_format(clust.main, add_idx=False)
        clust_head_lemma = coref_clean
        coref_id = coref_clean + '_' + str(main_mention.start) + str(main_mention.end)
        for ment in clust.mentions:
            ment_feats = [None] * 16
            ment_feats[10] = coref_id + '_cid'
            ment_feats[4] = ment.start  # assign start and end indices 
            ment_feats[5] = ment.end
            ment_feats[6] = ment.text  # mention literal form
            ment_feats[8] = clust_head_lemma
            # blindly assume cluster contain NNP, NNS, or NP 
            # TODO improve
            ment_feats[9] = 'NN'
            # blindy set doc, paragraph, and sent ids to zero 
            # TODO reconsider??
            feat_list[0] = 0
            feat_list[1] = 0
            feat_list[2] = 0
            # mention id
            feat_list[3] = _id_clean_format(feat_list[6])
            # head lemma
            feat_list[8] = _id_clean_format(feat_list[6])
            
            doc_features.append(ment_feats)
    
    return doc_features


def _extract_termite_feats(termite_xml, feat_list):
    """extract relevant features from termite xml and add to proper indices of feat list"""
    pass


def build_feat_list(nc_clusts, termite_xml):
    """take neural coref output and termite xml and build feature list (according to conll indexing)"""
    _extract_nc_feats(nc_clusts)
    _extract_termite_feats(termite_xml, feat_list)
    _calculate_missing_feats(feat_list)
