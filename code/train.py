import os
import random
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn

import csv
from metrics import calculate_metrics, get_mse

import warnings
warnings.filterwarnings("ignore")



def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)
    
def test(model, dataloader):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

        mol_vec = mol_vec.to(device)
        prot_vec = prot_vec.to(device)
        mol_mat = mol_mat.to(device)
        mol_mat_mask = mol_mat_mask.to(device)
        prot_mat = prot_mat.to(device)
        prot_mat_mask = prot_mat_mask.to(device)
        drugh_graph = drugh_graph.to(device).to(device)
        protein_graph = protein_graph.to(device).to(device)
        affinity = affinity.to(device)


        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse, ci, rm2 = calculate_metrics(labels, preds)
    return mse, ci, rm2

if __name__ == "__main__":
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Dataset-{hp.dataset}-{hp.running_set}") 
    print(f"Pretrain-{hp.mol2vec_dir}-{hp.protvec_dir}")
    save_metrics = {'mse':[], 'ci':[], 'rm2':[]}
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)
    
    train_dir = os.path.join(dataset_root, f'train.csv')
    valid_dir = os.path.join(dataset_root, f'valid.csv')
    test_dir = os.path.join(dataset_root, f'test.csv')                 
    train_set = CustomDataSet(pd.read_csv(train_dir), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir), hp)
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=12, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=12, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=12, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map))
    print("load dataset finished")
    
    model = nn.DataParallel(Model(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss
    
    train_log = []     
    best_valid_mse = 10  
    patience = 0    
    model_fromTrain = f'./GSIK-DTA/savemodel/{hp.dataset}-{hp.running_set}-{hp.current_time}.pth'
                 
    for epoch in range(1, hp.Epoch + 1):
        # trainning
        model.train()
        pred = []
        label = []             
        for batch_data in train_dataset_load:
            mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

            mol_vec = mol_vec.to(device)
            prot_vec = prot_vec.to(device)
            mol_mat = mol_mat.to(device)
            mol_mat_mask = mol_mat_mask.to(device)
            prot_mat = prot_mat.to(device)
            prot_mat_mask = prot_mat_mask.to(device)
            drugh_graph = drugh_graph.to(device).to(device)
            protein_graph = protein_graph.to(device).to(device)
            affinity = affinity.to(device)  
                  
            predictions = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
                
            loss = criterion(predictions.squeeze(), affinity)
            loss.backward()                
            optimizer.step()
            optimizer.zero_grad()                                             
        pred = np.array(pred)
        label= np.array(label)
        mse_value, ci_value, rm2_value = calculate_metrics(pred, label)
        train_log.append([mse_value, ci_value, rm2_value])
        print(f'Training Log at epoch: {epoch}: mse: {mse_value}')
            
        # valid
        mse, ci, rm2 = test(model, valid_dataset_load)   
        print(f'Valid at: mse: {mse}, ci: {ci}, rm2: {rm2}')
             
        # Early stop        
        if mse < best_valid_mse :
            patience = 0
            best_valid_mse = mse
            # save model
            torch.save(model.state_dict(), model_fromTrain)
            print(f'Update best_mse, Valid at epoch: {epoch}: mse: {mse}, ci: {ci}, rm2: {rm2}')
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
                break 
               
    log_dir = f"./GSIK-DTA/log/{hp.current_time}-{hp.dataset}-{hp.running_set}.csv"
    with open(log_dir, "w+")as f:
        writer = csv.writer(f)
        writer.writerow(["mse",  "ci", "rm2"])
        for r in train_log:
            writer.writerow(r)
    print(f'Save log over at {log_dir}')

    # Test
    predModel = nn.DataParallel(Model(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)    
    mse, ci, rm2 = test(predModel, test_dataset_load)
    print(f'Test at, mse: {mse}, ci: {ci}, rm2: {rm2}\n')
    save_metrics['mse'].append(mse)
    save_metrics['ci'].append(ci)
    save_metrics['rm2'].append(rm2)                             
        
        
    # save training log
    test_metrics = pd.DataFrame(save_metrics)    
    test_metrics.to_csv(f'./GSIK-DTA/log/Test-{hp.dataset}-{hp.running_set}-{hp.current_time}.csv', index=False)    
    mean_values = test_metrics.mean()
    variance_values = test_metrics.var()   
    print(f"Dataset-{hp.dataset}-{hp.running_set}")
    print(f"Mean Values:{pd.concat([mean_values, variance_values], axis=1)}")