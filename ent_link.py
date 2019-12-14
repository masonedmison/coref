import spacy
import scispacy

from scispacy.umls_linking import UmlsEntityLinker

nlp = spacy.load("en_core_sci_md")

# This line takes a while, because we have to download ~1GB of data
# and load a large JSON file (the knowledge base). Be patient!
# Thankfully it should be faster after the first time you use it, because
# the downloads are cached.
# NOTE: The resolve_abbreviations parameter is optional, and requires that
# the AbbreviationDetector pipe has already been added to the pipeline. Adding
# the AbbreviationDetector pipe and setting resolve_abbreviations to True means
# that linking will only be performed on the long form of abbreviations.
linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(linker)

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

# Let's look at a random entity!
entity = doc.ents[1]

print("Name: ", entity)

# Each entity is linked to UMLS with a score
# (currently just char-3gram matching).
for umls_ent in entity._.umls_ents:
	print(linker.umls.cui_to_entity[umls_ent[0]])
