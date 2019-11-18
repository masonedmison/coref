# Word Vector and NER Improvements  
> Currently running NeuralCoref from local copy with `ENTITY` added to `ACCEPTED_ENTS`
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

### Possible SpaCy moves:
- Rule based matching - Entity Ruler <https://spacy.io/usage/rule-based-matching#entityruler>
    - use ontologies to create jsonld file to annotate genes/ proteins -- add patterns to Entity Ruler
- Entity Linking? More research needed... 

### Isolated SciSpacy Word Vectors
- Looks as thought neural coref DOES use word vectors within spacy, if not given as **cfg_inference 'conv_dict'=val
    - review spacy code and documentation to see how we could integrate other vectors in (ElMO, BioWordVec, BioBert, etc.)

## TODO
- experiment with NER SciSpacy models (which are NER models trained on top of other models, e.g. md_core_sci_md) 

