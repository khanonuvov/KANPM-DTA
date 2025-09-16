from tqdm import tqdm
import pickle
import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaModel

def get_chem_pretrain(df_dir, db_name, max_smiles_length=220):
  
    # Load and prepare data
    df = pd.read_excel(df_dir)
    
    # Initialize model
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    # Storage for embeddings
    embeddings = {
        "dataset": db_name,
        "vec_dict": {},
        "mat_dict": {},
        "length_dict": {} 
    }

    # Process each SMILES string
    for drug_id, smile in tqdm(zip(df['drug_id'], df['drug_seq']), 
                              total=len(df), desc="Processing SMILES"):
        drug_id = str(drug_id)
        smile = smile[:max_smiles_length]  # Truncate if needed

        with torch.no_grad():
            outputs= model(**tokenizer(smile, return_tensors='pt'))
            embeddings_out = outputs.last_hidden_state[0][1:outputs.last_hidden_state.shape[1]-1]
            
        reps = embeddings_out
        
        # Store embeddings
        embeddings["mat_dict"][drug_id] = reps
        embeddings["vec_dict"][drug_id] = reps.mean(axis=0)
        embeddings["length_dict"][drug_id] = len(smile)
    
    # Save embeddings
    output_path = f'./KANPM-DTA/pretrained/{db_name}/{db_name}_chem_pretrained.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved embeddings for {len(embeddings['vec_dict'])} compounds to {output_path}")

db_names = ['davis', 'kiba', 'metz']
df_dirs = [r'./KANPM-DTA/datasets/davis/davis_drugs.csv', r'./KANPM-DTA/datasets/kiba/kiba_drugs.csv', r'./KANPM-DTA/datasets/metz/metz_drugs.csv']

for i in range(0,3):
    print(f'Compute {df_dirs[i]} drug pretrain feature by Chemberta-2.')

    get_esm_contact_map(model, df_dirs[i], db_names[i])
