from lxml import etree as et 
import requests
import os
import sys


def get_termite_xml(text, format_='any.xml', output='xml'):
    url = 'http://termite:8080/termite'
    form_data = {
        'text': text,
        'format': format_,
        'output': output
    }
    response = requests.get(url, params=form_data)
    return response.content


def get_concept_mappings(text, relevant=("DRUG", "DRUGFIND", "INDICATION", "GENE", "CHEMREC", "GENEBOOST", "DRUGTYP", "PROTYP")):
    xml = get_termite_xml(text)
    tree = et.ElementTree(et.fromstring(xml))
    root = tree.getroot()
    hits = root.iter("Hit")
    output = {}
    for h in hits:
        if h.attrib["type"] in relevant:
            concept = h.find("Name").text
            synonyms = [s.text for s in h.iter("MatchedSynonym")]
            output.update({concept: synonyms})
    return output


def get_rel_types(text, relevant=("DRUG", "DRUGFIND", "INDICATION", "GENE", "CHEMREC", "GENEBOOST", "DRUGTYP", "PROTYP")):
    xml = get_termite_xml(text)
    tree = et.ElementTree(et.fromstring(xml))
    root = tree.getroot()
    hits = root.iter("Hit")
    output = set()
    for h in hits:
        if h.attrib["type"].strip() in relevant:
            concept = h.find("Name").text
            all_lits = {s.text.lower() for s in h.iter("MatchedSynonym")}
            all_lits.add(concept.lower())
            output = output.union(all_lits)
    return output
