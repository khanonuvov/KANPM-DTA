import os
import numpy as np 
import pandas as pd 
import pickle
import random
from tqdm import tqdm
from rdkit import Chem
from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, pred_my_collate_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from metrics import calculate_metrics

import warnings
warnings.filterwarnings("ignore")


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)
    
def test(model, dataloader):
    model.eval()
    preds = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, drugh_graph, protein_graph = batch_data


        mol_vec = mol_vec.to(device)
        prot_vec = prot_vec.to(device)
        mol_mat = mol_mat.to(device)
        mol_mat_mask = mol_mat_mask.to(device)
        prot_mat = prot_mat.to(device)
        prot_mat_mask = prot_mat_mask.to(device)
        drugh_graph = drugh_graph.to(device).to(device)
        protein_graph = protein_graph.to(device).to(device)

        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()

    preds = np.array(preds)

    return preds


SEED = 0 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)

hp = HyperParameter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# pretrained model pth 
model_fromTrain =  './KANPM-DTA/savemodel/saved_model.pth'


drug_df = pd.read_csv(hp.drugs_dir)
prot_df = pd.read_csv(hp.prots_dir)
mol2vec_dict = load_pickle(hp.mol2vec_dir)
protvec_dict = load_pickle(hp.protvec_dir)
contact_map = load_pickle(hp.contact_map)


test_dir = './KANPM-DTA/datasets/test/test.csv'
test_df = pd.read_csv(test_dir)
test_set = CustomDataSet(test_df, hp)
test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=False, num_workers=8, collate_fn=lambda x: pred_my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
 

predModel = nn.DataParallel(Model(hp, device))
predModel.load_state_dict(torch.load(model_fromTrain))
predModel = predModel.to(device)    
preds = test(predModel, test_dataset_load)
print('preds: ', preds)
print('Predictions generated!')

assert len(test_df) == len(preds), "Error: Length mismatch!"

results_df = pd.DataFrame({
    'drug_id': test_df['drug_key'],
    'drug_names': test_df['drug_names'],
    'protein_id': test_df['target_key'],
    'prediction': preds
})

output_csv = './KANPM-DTA/preds.csv'
results_df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")


