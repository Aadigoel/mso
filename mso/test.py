#from mso.mso.objectives.mol_functions import solubility
import sys
sys.path.append("C:\Workspaces\iiith\mso\mso")

from mso.objectives.mol_functions import logp_score
from IPython.display import display
from mso.objectives.mol_functions import tox_alert
from mso.optimizer import BasePSOptimizer
from mso.objectives.scoring import ScoringFunction
from mso.objectives.mol_functions import qed_score, substructure_match_score, penalize_macrocycles, heavy_atom_count#, heavy_molecular_weight
import mso.objectives.mol_functions as molf
from cddd.inference import InferenceModel
from rdkit import Chem
from functools import partial


infer_model = InferenceModel("C:\Workspaces\iiith\cddd\default_model") # The CDDD inference model used to encode/decode molecular SMILES strings to/from the CDDD space. You might need to specify the path to the pretrained model (e.g. default_model)
init_smiles = "c1ccccc1" # SMILES representation of benzene
hac_desirability = [{"x": 0, "y": 0}, {"x": 5, "y": 0.1}, {"x": 15, "y": 0.9}, {"x": 20, "y": 1.0}, {"x": 25, "y": 1.0}, {"x": 30, "y": 0.9,}, {"x": 40, "y": 0.1}, {"x": 45, "y": 0.0}]
substructure_match_score = partial(substructure_match_score, query=Chem.MolFromSmiles("c1ccccc1")) # use partial to define the additional argument (the substructure) 
miss_match_desirability = [{"x": 0, "y": 1}, {"x": 1, "y": 0}] # invert the resulting score to penalize for a match.
scoring_functions = [
    # ScoringFunction(molf.heavy_molecular_weight, "hmw", desirability=hac_desirability, is_mol_func=True),
    #ScoringFunction(heavy_atom_count, "hac", desirability=hac_desirability, is_mol_func=True),
    ScoringFunction(qed_score, "qed", is_mol_func=True),
    ScoringFunction(substructure_match_score, "miss_match",desirability=miss_match_desirability, is_mol_func=True),
    ScoringFunction(logp_score, "partition", is_mol_func=True),
    ScoringFunction(tox_alert, "mol_sol", is_mol_func=True),
    ScoringFunction(penalize_macrocycles, "pen_macro", is_mol_func=True)
]
opt = BasePSOptimizer.from_query(
    init_smiles=init_smiles,
    num_part=50,
    num_swarms=1,
    inference_model=infer_model,
    scoring_functions=scoring_functions)

opt.run(5)
display(opt.best_solutions)