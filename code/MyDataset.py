import torch
import os.path as osp
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data, Batch
import networkx as nx
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def matrix_pad_drug(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = arr
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = arr[:max_len]
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask

def matrix_pad_prot(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = torch.from_numpy(arr)
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = torch.from_numpy(arr[:max_len])
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask


def target2graph(distance_map, protein_features_esm):
    target_edge_index = []
    target_edge_distance = []
    protein_features_esm = protein_features_esm[1:-1, :]
    target_size = protein_features_esm.shape[0]    

    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    index_row, index_col = np.where(distance_map >= 0.5)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_distance.append(distance_map[i, j])

    target_feature = torch.FloatTensor(protein_features_esm)
    target_edge_index = torch.LongTensor(target_edge_index).transpose(1, 0)
    target_edge_distance = torch.FloatTensor(target_edge_distance)

    return target_size, target_feature, target_edge_index, target_edge_distance

def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                'Pt', 'Hg', 'Pb', 'X']]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['degree'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['ImplicitValence'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['num_h'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['hybridization'] == x) for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))


    return edge_index, edge_attr

def smile2graph(smile):
    mol = Chem.MolFromSmiles(smile)

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    mol_size = mol.GetNumAtoms()
    g = nx.DiGraph()
    
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                a_type=atom_i.GetSymbol(),
                a_num=atom_i.GetAtomicNum(),
                acceptor=0,
                donor=0,
                aromatic=atom_i.GetIsAromatic(),
                hybridization=atom_i.GetHybridization(),
                num_h=atom_i.GetTotalNumHs(),
                degree = atom_i.GetDegree(),
                # 5 more node features
                ExplicitValence=atom_i.GetExplicitValence(),
                FormalCharge=atom_i.GetFormalCharge(),
                ImplicitValence=atom_i.GetImplicitValence(),
                NumExplicitHs=atom_i.GetNumExplicitHs(),
                NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
            )
            
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor']

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                      
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
                
    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)         

    return mol_size, node_attr, edge_index, edge_attr


def my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    b_label = torch.zeros(batch_size, dtype=torch.float32)
    
    b_drug_graph = []    
    b_protein_graph = []
    
    # Process each sample in the batch
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id, label = pair[0], pair[2], pair[4]
        drug_smiles = drug_df.loc[drug_df['drug_key'] == drug_id, 'compound_iso_smiles'].iloc[0]
        prot_seq = prot_df.loc[prot_df['target_key'] == prot_id, 'target_sequence'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_vec = mol2vec_dict["vec_dict"][drug_id]
        prot_vec = protvec_dict["vec_dict"][prot_id]
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)        
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max) 

        # Drug graph for PyTorch Geometric
        mol_size, node_attr, edge_index, edge_attr = smile2graph(drug_smiles)
        drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)
        
        target_size, target_features, target_edge_index, target_edge_distance = target2graph(prot_contact_map, prot_mat)
        protein_graph = Data(x=target_features, edge_index=target_edge_index, edge_weight=target_edge_distance)
        b_protein_graph.append(protein_graph)
        
        
        # Store other values for the batch
        b_drug_vec[i] = drug_vec
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
        b_label[i] = label
    
    # Batch graphs using PyG's built-in functionality
    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_drug_graph, b_protein_graph, b_label

def pred_my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    
    b_drug_graph = []    
    b_protein_graph = []
    
    # Process each sample in the batch
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id = pair[0], pair[2]
        drug_smiles = drug_df.loc[drug_df['drug_key'] == drug_id, 'compound_iso_smiles'].iloc[0]
        prot_seq = prot_df.loc[prot_df['target_key'] == prot_id, 'target_sequence'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_vec = mol2vec_dict["vec_dict"][drug_id]
        prot_vec = protvec_dict["vec_dict"][prot_id]
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)        
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max) 

        # Drug graph for PyTorch Geometric
        mol_size, node_attr, edge_index, edge_attr = smile2graph(drug_smiles)
        drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)
        
        target_size, target_features, target_edge_index, target_edge_distance = target2graph(prot_contact_map, prot_mat)
        protein_graph = Data(x=target_features, edge_index=target_edge_index, edge_weight=target_edge_distance)
        b_protein_graph.append(protein_graph)
        
        
        # Store other values for the batch
        b_drug_vec[i] = drug_vec
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
    
    # Batch graphs using PyG's built-in functionality
    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_drug_graph, b_protein_graph


class CustomDataSet(Dataset):
    def __init__(self, dataset, hp):    
        self.hp = hp
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset.iloc[index,:]

    def __len__(self):
        return len(self.dataset)
