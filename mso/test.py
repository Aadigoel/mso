#from mso.mso.objectives.mol_functions import solubility
import sys
from io import StringIO
import json
sys.path.append("C:\Workspaces\iiith\mso\mso")
import matplotlib.pyplot as plt
import numpy as np
from mso.objectives.mol_functions import logp_score
from IPython.display import display
from mso.objectives.mol_functions import tox_alert
from mso.optimizer import BasePSOptimizer
from mso.objectives.scoring import ScoringFunction
from mso.objectives.mol_functions import qed_score, substructure_match_score, penalize_macrocycles
from cddd.inference import InferenceModel
from rdkit import Chem
from functools import partial
import warnings
from mso.data import data_dir
import os
import pandas as pd
import numpy as np
from functools import wraps
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import RDConfig
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit import DataStructs
import networkx as nx
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


import streamlit as st


st.set_page_config(page_title='D4: MolOpt')
st.title('Molecule Optimization Dashboard')
smarts = pd.read_csv(os.path.join(data_dir, "sure_chembl_alerts.txt"), header=None, sep='\t')[1].tolist()
alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]


def check_valid_mol(func):
    """
    Decorator function that checks if a mol object is None (resulting from a non-processable SMILES string)
    :param func: the function to decorate.
    :return: The decorated function.
    """
    @wraps(func)
    def wrapper(mol, *args, **kwargs):
        if mol is not None:
            return func(mol, *args, **kwargs)
        else:
            return 0
    return wrapper

@check_valid_mol
def heavy_molecular_weight(mol): 
    """heavy molecule weight"""
    hmw = Chem.Descriptors.HeavyAtomMolWt(mol)
    return hmw

def sa_score(mol):
    """
    Synthetic acceptability score as proposed by Ertel et al..
    """
    try:
        score = sascorer.calculateScore(mol)
    except:
        score = 0
    return score

@check_valid_mol
def logp_score(mol):
    """
    crippen logP
    """
    score = Chem.Crippen.MolLogP(mol)
    return score

@check_valid_mol
def neg_simil(mol):
    """Negative reward if similar to previous best molecule"""

fitness_history = []
def run(num_steps, num_track=10):
        """
        The main optimization loop.
        :param num_steps: The number of update steps.
        :param num_track: Number of best solutions to track.
        :return: The optimized particle swarm.
        """
        # evaluate initial score
        pbar = st.progress(0)
        display_smiles = st.text("Generating SMILES...")
        display_reward = st.text("Running..")
        display_plot = st.text("Plotting..")
        for swarm in opt.swarms:
            opt.update_fitness(swarm)
        
        for step in range(num_steps):
            opt._update_best_fitness_history(step)
            max_fitness, min_fitness, mean_fitness = opt._update_best_solutions(num_track)
            # print("Step %d, max: %.3f, min: %.3f, mean: %.3f"
            #       % (step, max_fitness, min_fitness, mean_fitness))
            pbar.progress(int(100/num_steps)*(step+1))
            display(opt.best_fitness_history.iloc[-1])
            opt.best_fitness_history.astype(str)
            smile_st = str(opt.best_fitness_history.iloc[-1]["smiles"])
            smile_mol = Chem.MolFromSmiles(smile_st)
            smile_image = Draw.MolToImage(smile_mol)
            display_smiles.image(smile_image)
            fitness = str(opt.best_fitness_history.iloc[-1]["fitness"])
            display_reward.text("Reward: "+ str(float(fitness)*100))
            fitness_history.append(round(float(fitness)*100,2))


            chart_data = pd.DataFrame(fitness_history)
            fig, ax = plt.subplots()
            plt.xlim(1,num_steps)
            plt.ylim(0,100)
            plt.xlabel("Num Steps")
            plt.ylabel("Reward")
            ax.figure.set_size_inches(3.2,2.4)
            ax.plot([i+1 for i in range(len(fitness_history))],fitness_history)
            display_plot.pyplot(fig)            

            
            for swarm in opt.swarms:
                opt._next_step_and_evaluate(swarm)
            
            
        return opt.swarms

infer_model = InferenceModel("C:\Workspaces\iiith\cddd\default_model") # The CDDD inference model used to encode/decode molecular SMILES strings to/from the CDDD space. You might need to specify the path to the pretrained model (e.g. default_model)
init_smiles = "c1ccccc1" # SMILES representation of benzene
hac_desirability = [{"x": 0, "y": 0}, {"x": 5, "y": 0.1}, {"x": 15, "y": 0.9}, {"x": 20, "y": 1.0}, {"x": 25, "y": 1.0}, {"x": 30, "y": 0.9,}, {"x": 40, "y": 0.1}, {"x": 45, "y": 0.0}]
substructure_match_score = partial(substructure_match_score, query=Chem.MolFromSmiles("c1ccccc1")) # use partial to define the additional argument (the substructure) 
miss_match_desirability = [{"x": 0, "y": 1}, {"x": 1, "y": 0}]
file_json = st.file_uploader("Upload a JSON file", type=([".json"]))
num_steps = int(st.number_input("Numer of Iterations: ",min_value=10, max_value=30))
if file_json:
    stringio = StringIO(file_json.getvalue().decode("utf-8"))
    string_data = stringio.read()
    dict_data = json.loads(string_data)
    st.write(dict_data)

func_name = st.text_input("Enter name of function")

custom = st.text_area("Enter text")

submit = st.button("Submit")

if submit:
    # custom = lambda x: None
    # lambda_list = []
    # if custom_func:
    #     for func in custom_funcs:
    #         lambda_list.append(eval(func))
    #     #custom = eval(custom_func)
    scoring_functions = []
    custom_eval = exec(custom)

    for i in dict_data["parameters"]:
        if i["name"]== "qed_score":
            scoring_functions.append(ScoringFunction(qed_score, "qed", weight= i["weight"], is_mol_func=True))
        if i["name"]== "heavy_molecular_weight":
            scoring_functions.append(ScoringFunction(heavy_molecular_weight, "hmw", desirability=hac_desirability, is_mol_func=True))
        if i["name"]== "substructure_match_score":
            scoring_functions.append(ScoringFunction(substructure_match_score, "miss_match",desirability=miss_match_desirability, is_mol_func=True))
        if i["name"]== "logp_score":
            scoring_functions.append(ScoringFunction(logp_score, "partition", weight=i["weight"], is_mol_func=True))
        if i["name"]== "tox_alert":
            scoring_functions.append(ScoringFunction(tox_alert, "mol_sol", is_mol_func=True))
        if i["name"]== "penalize_macrocycles":
            scoring_functions.append(ScoringFunction(penalize_macrocycles, "pen_macro", is_mol_func=True))
        if i["name"]== "sa_score":
            scoring_functions.append(ScoringFunction(sa_score, "sa_score", weight=i["weight"], is_mol_func= True))
        if i["name"] == "custom":
            scoring_functions.append(ScoringFunction(eval(func_name), "Custom", weight = i["weight"], is_mol_func = True))

    opt = BasePSOptimizer.from_query(
        init_smiles=init_smiles,
        num_part=10,
        num_swarms=1,
        inference_model=infer_model,
        scoring_functions=scoring_functions)

    run(num_steps)

    st.balloons()
    print(opt.best_solutions)
    df = pd.DataFrame(columns = ["SMILES", "QED", "SAS", "LogP", "Tox_alert"])
    st.text("Top Hits:")
    for i in range(10):
        smile_st = str(opt.best_solutions.iloc[i]["smiles"])
        mol_st = Chem.MolFromSmiles(smile_st)
        caption = round(float(opt.best_solutions.iloc[i]["fitness"])*100,2)
        hmw = heavy_molecular_weight(mol_st)
        qed = round(float(Chem.Descriptors.qed(mol_st)), 2)
        tox = tox_alert(mol_st)
        sas = sa_score(mol_st)
        miss_match = substructure_match_score(mol_st)
        partition = round(float(logp_score(mol_st)),2)
        pen_macro = penalize_macrocycles(mol_st)
        smile_image = Draw.MolToImage(mol_st)
        df.loc[i] = [smile_st, qed, sas, partition, tox]
        st.image(smile_image,caption="Reward: "+str(caption)+"\nQED: "+str(qed)+" LogP:"+str(partition))
    
    print(df)
       
#display(opt.best_solutions)