## Overview of explorations and Possible Directions 11-9/11-18

### SciSpacy Integration
- Scispacy is currently integrated by way of including `ENTITY` in `ACCEPTED_ENTS` constant

#### Regarding NER Modules
- NER modules trained on top of `md` piplines available - have not run on dataset (problems were occuring and was kicking out after only processing a few abtracts)
#### Performance/ Eval Results
- more clusters generated
- performs slightly worse than the general spacy model - due to larger number of false positives


### Seperate eval of mentions and clusters
- Rainer mentioned the possibility of seperating problem into two sub-tasks: Entity Recognition and Coreference Resolution
    - Possiblity of examining stats of annotations, i.e. Anaphor is Relative Pronoun, Pronoun, Definite Noun Phrase,etc. and Antecedent Includes Conjunction, Cross-Sentence, Identical Relation, Including Protein, etc. 
    - Examine how well SciSpacy NER module is performing
    - How well does neural coref perform if the NER module identifies **all** mentions

### Vector Integration
- different use of vectors (appears) to be possible by way of integration through SpaCY/ SciSpaCY
    - maybe a 4-6 hour task to build in different vectors in a way that is reproducible


### general evaluation script tweaks
- compare performance difference between chaining and distributative? 
