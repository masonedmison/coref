# Word Vector and NER Improvements  

## SciSpacy 
### As Full Pipeline
#### Pros:
- Biomedical word vector also ship with full SciSpacy Model
- Domain Specific NER module 
#### Cons:
- NER demarcates **all** entities as `ENTITY` label. Neural Coref does not include this entity in the `Rule-Based-Matching` constants

```python 
##### STRINGS USED IN RULE_BASED MENTION DETECTION #######

NO_COREF_LIST = ["i", "me", "my", "you", "your"]
MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}
KEEP_TAGS = ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "DT", "IN"]
CONTENT_TAGS = ["NN", "NNS", "NNP", "NNPS"]
PRP_TAGS = ["PRP", "PRP$"]
CONJ_TAGS = ["CC", ","]
PROPER_TAGS = ["NNP", "NNPS"]
NSUBJ_OR_DEP = ["nsubj", "dep"]
CONJ_OR_PREP = ["conj", "prep"]
LEAVE_DEP = ["det", "compound", "appos"]
KEEP_DEP = ["nsubj", "dobj", "iobj", "pobj"]
REMOVE_POS = ["CCONJ", "INTJ", "ADP"]
LOWER_NOT_END = ["'s", ',', '.', '!', '?', ':', ';']
PUNCTS = [".", "!", "?"]
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]

##########################################################
```
### Isolated SciSpacy Word Vectors

### Isolated NER Module

## BioWordVec
