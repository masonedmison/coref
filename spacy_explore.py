import spacy
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.abbreviation import AbbreviationDetector


# pub_abs = """Leukotriene B4 stimulates c-fos and c-jun gene transcription and AP-1 binding activity in human monocytes.\n
# We have examined the effect of leukotriene B4 (LTB4), a potent lipid proinflammatory mediator, on the expression of the\n
# proto-oncogenes c-jun and c-fos."""

#
nlp = spacy.load('en_core_sci_md')
# linker = UmlsEntityLinker(resolve_abbreviations=True)

# nlp.add_pipe(linker)

doc = nlp(u"""Leukotriene B4 stimulates c-fos and c-jun gene transcription and AP-1 binding activity in human monocytes.
We have examined the effect of leukotriene B4 (LTB4), a potent lipid proinflammatory mediator, on the expression of the
proto-oncogenes c-jun and c-fos.""")



print(dir(doc.ents[0]))
for ent in doc.ents:
    # print(f'[ENT] {ent.text}, starts at {ent.start} and ends at {ent.end}')
    print(f'ENT {ent}\nLABEL {ent.label_}')
# abbreviations
# print("Abbreviation", "\t", "Definition")
# for abrv in doc._.abbreviations:
# 	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

# for token in doc:
#     print(f'[FOR WORD] {token.text}\n')
#     print(f'{token.text}, {token.lemma_},\n \t[PART OF SPEECH]{token.pos_},\n \t[TAG] {token.tag_},\n \t[DEP]{token.dep_}')
#     print('-'*35)
