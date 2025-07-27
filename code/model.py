import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import HyperParameter
from collections import OrderedDict
import math
from kan import KAN
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedFusionLayer(nn.Module):
    def __init__(self, v_dim, q_dim, output_dim=128, dropout_rate=0.2):
        super(GatedFusionLayer, self).__init__()
        self.v_transform = nn.Linear(v_dim, output_dim)
        self.q_transform = nn.Linear(q_dim, output_dim)
        self.gate_transform = nn.Linear(output_dim*2, output_dim)
        self.activation = nn.Tanh()
        self.output_dim = output_dim

    def get_output_shape(self):
        return self.output_dim

    def forward(self, v, q):
        v_proj = self.activation(self.v_transform(v))
        q_proj = self.activation(self.q_transform(q))

        concat_proj = torch.cat([v_proj, q_proj], dim=1)
        gate = torch.sigmoid(self.gate_transform(concat_proj))

        gated_output = gate * v_proj + (1 - gate) * q_proj
        return gated_output


class DrugGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=88,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(DrugGraphNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)


        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)


        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight.mean(dim=1)))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class ProteinGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=1152,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(ProteinGraphNet, self).__init__()

        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class LinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, heads=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.linear_first = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.heads)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, masks):
        sentence_att = F.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = sentence_att.transpose(1, 2)
        minus_inf = -9e15 * torch.ones_like(sentence_att)
        e = torch.where(masks > 0.5, sentence_att, minus_inf)
        att = self.softmax(e)
        sentence_embed = att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.heads

        return avg_sentence_embed

class MODEL(nn.Module):

    def __init__(self, hp, device):
        super(MODEL, self).__init__()

        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim
        self.encoder_layers = 3
        self.encoder_heads = 8
        self.feedforward_dim = 1024

        self.dropout = 0.2

        self.drug_graph_model = DrugGraphNet(n_output=128)
        self.protein_graph_model = ProteinGraphNet(n_output=128)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=self.encoder_layers)

        self.drug_attn = LinearAttention(128, 64, 8)
        self.target_attn = LinearAttention(128, 64, 8)
        self.inter_attn_one = LinearAttention(128, 64, 8)

        self.drug_ln = nn.LayerNorm(128)
        self.target_ln = nn.LayerNorm(128)
        
        self.attnention = GatedFusionLayer(v_dim=128, q_dim=128, output_dim=128, dropout_rate=0.2)        

        self.fc2 = nn.Linear(self.protvec_dim, 128)
        self.fc3 = nn.Linear(self.mol2vec_dim, 128)

        self.kan_lin =  KAN([512, 1024, 512, 1])

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, drug_graph, protein_graph):

        smiles_graph = self.drug_graph_model(drug_graph)        
        fasta_graph = self.protein_graph_model(protein_graph)

        smiles_emb = self.transformer_encoder(self.fc3(drug_mat))
        xd = self.drug_ln(smiles_emb)
        smiles_mask = self.generate_masks(xd, 128, 8)
        xd_attn = self.drug_attn(xd, smiles_mask)

        fasta_emb = self.transformer_encoder2(self.fc2(prot_mat))
        xp = self.target_ln(fasta_emb)
        fasta_mask = self.generate_masks(xp, 128, 8)
        xp_attn = self.target_attn(xp, fasta_mask)

        cat_f = torch.cat([xp, xd], dim=1)
        cat_mask = torch.cat([fasta_mask, smiles_mask], dim=-1)
        cat_attn = self.inter_attn_one(cat_f, cat_mask)

        graph_att = self.attnention(smiles_graph, fasta_graph)

        out = self.kan_lin(torch.cat([xd_attn, cat_attn, xp_attn, graph_att], dim=-1))
        return out
