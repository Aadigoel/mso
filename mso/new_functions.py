"""
Module with scoring functions that take RDKit mol objects as input for scoring.
"""
import warnings
from mso.data import data_dir
import os
import pandas as pd
import numpy as np
from functools import wraps
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit import DataStructs
import networkx as nx


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