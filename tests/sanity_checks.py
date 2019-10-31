"""Contains various Sanity Checks"""
import spacy
import neuralcoref
from bionlp_eval import cluster, span_

# text examples to use 
text1 = """Redox status of cells influences constitutive or induced NF-kappa B translocation and HIV long terminal repeat activity in human T and monocytic cell lines. 
We have tested the hypothesis that cellular activation events occurring in T lymphocytes and monocytes and mediated through translocation of the transcription factor NF-kappa B are dependent upon the constitutive redox status of these cells. We used phenolic, lipid-soluble, chain-breaking antioxidants (butylated hydroxyanisole (BHA), nordihydroquairetic acid, or alpha-tocopherol (vitamin E) to show that peroxyl radical scavenging in unstimulated and PMA- or TNF-stimulated cells blocks the functions depending on NF-kappa B activation. BHA was found to suppress not only PMA- or TNF-induced, but also constitutive, HIV-enhancer activity concomitant to an inhibition of NF-kappa B binding activity in both lymphoblastoid T (J.Jhan) and monocytic (U937) cell lines. This was also true for KBF (p50 homodimer) binding activity in U937 cells. Secretion of TNF, the product of another NF-kappa B-dependent gene, was abolished by BHA in PMA-stimulated U937 cells. The anti-oxidative effect of BHA was accompanied by an increase in thiol, but not glutathione, content in stimulated and unstimulated T cell, whereas TNF stimulation itself barely modified the cellular thiol level. Oxidative stress obtained by the addition of H2O2 to the culture medium of J.Jhan or U937 cells could not by itself induce NF-kappa B activation. These observations suggest that TNF and PMA do not lead to NF-kappa B activation through induction of changes in the cell redox status. Rather, TNF and PMA can exert their effect only if cells are in an appropriate redox status, because prior modification toward reduction with BHA treatment prevents this activation. It appears that a basal redox equilibrium tending toward oxidation is a prerequisite for full activation of transduction pathways regulating the activity of NF-kappa B-dependent genes. """

# T2	Exp 173 187	the hypothesis
# T3	Exp 188 192	that
# T4	Exp 1270 1285	TNF stimulation	1274 1285	stimulation
# T5	Exp 1286 1292	itself
# T6	Exp 1335 1430	Oxidative stress obtained by the addition of H2O2 to the culture medium of J.Jhan or U937 cells	1335 1351	Oxidative stress
# T7	Exp 1444 1450	itself
# T8	Exp 1625 1636	TNF and PMA
# T9	Exp 1647 1652	their
# R1	Coref Anaphora:T3 Antecedent:T2
# R2	Coref Anaphora:T5 Antecedent:T4
# R3	Coref Anaphora:T7 Antecedent:T6
# R4	Coref Anaphora:T9 Antecedent:T8




nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, greedyness=0.5)
# ------------------------------------------
# word index vs char index
doc = nlp(text1)
def wordInd_to_charInd_check():



# ------------------------------------------