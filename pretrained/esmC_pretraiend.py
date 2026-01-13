# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:49:30 2025
@author: Yoush
"""
import pickle
from tqdm import tqdm
import pandas as pd
import torch

# Import ESMC model and ESMProtein wrapper
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Load model
model = ESMC.from_pretrained("esmc_600m").to("cuda").eval()

# Set max sequence length
MAX_SEQ_LENGTH = 1200  # ESM-C supports long sequences

def get_esmc_pretrain(model, df_dir, db_name):
    df = pd.read_excel(df_dir)
    
    emb_dict, emb_mat_dict, length_target = {}, {}, {}

    for prot_id, seq in tqdm(zip(df['prot_id'], df['prot_seq']), total=len(df)):
        seq = seq[:MAX_SEQ_LENGTH]  # truncate if needed
        protein = ESMProtein(sequence=seq)

        with torch.no_grad():
            # Use LogitsConfig to request embeddings
            logits_output = model.logits(
                model.encode(protein),
                LogitsConfig(sequence=True, return_embeddings=True)
            )

        reps = logits_output.embeddings[0]
        reps = reps.cpu().numpy()
        
        emb_mat_dict[prot_id] = reps
        emb_dict[prot_id] = reps.mean(axis=0)
        length_target[prot_id] = len(seq)

    # Save embeddings to pickle
    with open(f'./KANPM-DTA/pretrained/{db_name}/{db_name}_esmc_pretrain.pkl', 'wb') as f:
        pickle.dump({
            "dataset": db_name,
            "vec_dict": emb_dict,
            "mat_dict": emb_mat_dict,
            "length_dict": length_target
        }, f)
    
    print(f"✅ Saved {db_name} ESM-C features.")


db_names = ['davis', 'kiba', 'metz']
df_dirs = [r'./KANPM-DTA/datasets/davis/davis_prots.csv', r'./KANPM-DTA/datasets/kiba/kiba_prots.csv', r'./KANPM-DTA/datasets/metz/metz_prots.csv']

for i in range(0,3):
    print(f'Compute {df_dirs[i]} protein pretrain feature by esm-c.')

    get_esmc_pretrain(model, df_dirs[i], db_names[i])

