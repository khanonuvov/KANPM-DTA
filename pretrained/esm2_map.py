# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:49:30 2025

@author: Yoush
"""

import pickle
from tqdm import tqdm
import pickle
import esm
import pandas as pd
import torch


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)


model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def get_esm_contact_map(model, df_dir, db_name, max_length=1200):
    df = pd.read_csv(df_dir)    
    prot_data = []
    for prot_id, seq in zip(df['prot_id'], df['prot_seq']):
        truncated_seq = seq[:max_length]
        prot_data.append((prot_id, truncated_seq))
    
    target_graph = {}
    length_target = {}

    for prot_id, seq in tqdm(prot_data):
        batch_labels, batch_strs, batch_tokens = batch_converter([(prot_id, seq)])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        contact_map = results["contacts"][0].numpy()

        target_graph[prot_id] = contact_map
        length_target[prot_id] = len(seq)
    
    dump_data = {
        "dataset": db_name,
        "contact_map": target_graph,
        "length_dict": length_target
    }
    
    output_path = f'./KANPM-DTA/pretrained/{db_name}/{db_name}_esm2_contact_map.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(dump_data, f)

    print(f"✅ Saved contact maps for {db_name}.")
        

db_names = ['davis', 'kiba', 'metz']
df_dirs = [r'./KANPM-DTA/datasets/davis/davis_prots.csv', r'./KANPM-DTA/datasets/kiba/kiba_prots.csv', r'./KANPM-DTA/datasets/metz/metz_prots.csv']

for i in range(0,3):
    print(f'Compute {df_dirs[i]} protein pretrain feature by esm2.')
    get_esm_contact_map(model, df_dirs[i], db_names[i])
    

