import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math
import torch.nn as nn

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
    def __init__(self, handler):
        super(Model, self).__init__()
        
        self.dEmbeds = nn.Parameter(init(torch.empty(args.num_drug, args.latdim)))
        self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))
        self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)

        self.kg_dict = handler.kg_dict
        self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)

        self.torchDDIAdj = handler.torchDDIAdj

        self.sigmoid = nn.Sigmoid()

        self.args = args
        
        self.leakyrelu = nn.LeakyReLU()

                
    def forward(self, mess_dropout=True):
        hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)         
        kg_d_embeds = hids_KG[:args.num_drug, :]
        
        embedsLst = [self.dEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(self.torchDDIAdj, embedsLst[-1])
            embedsLst.append(embeds)
        ddi_d_embeds = sum(embedsLst)

        return ddi_d_embeds, kg_d_embeds
    
    def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
        sampleEdges = []
        for h in kg_dict:
            t_list = kg_dict[h]
            if triplet_num != -1 and len(t_list) > triplet_num:
                sample_edges_i = random.sample(t_list, triplet_num)
            else:
                sample_edges_i = t_list
            for r, t in sample_edges_i:
                sampleEdges.append([h, t, r])
        return self.getEdges(sampleEdges)
    
    def getEdges(self, kg_edges):
        graph_tensor = torch.tensor(kg_edges)
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long(), type.long()


    def get_rating(self, emb, input1, input2, de_data):

        ddi_d_embeds = emb[0]
        kg_d_embeds = emb[1]

        drug1_ddi_emb = ddi_d_embeds[input1.long()]
        drug2_ddi_emb = ddi_d_embeds[input2.long()]

        drug1_kg_emb = kg_d_embeds[input1.long()]
        drug2_kg_emb = kg_d_embeds[input2.long()]

        drug1_d_kg_emb = de_data[0]
        drug2_d_kg_emb = de_data[1]


        drug1_emb = drug1_ddi_emb + drug1_d_kg_emb
        drug2_emb = drug2_ddi_emb + drug2_d_kg_emb

        pre_result1 = self.sigmoid(torch.sum(drug1_emb * drug2_emb, 1))

        return pre_result1


    def get_test_rating(self, drug_embeddings, drug1, drug2, molecule_embeddings, entity_embeddings, DiffProcess, KGDNet):
        
        ddi_d_embeds = drug_embeddings[0]
        kg_d_embeds = drug_embeddings[1]

        drug1_ddi_emb = ddi_d_embeds[drug1.long()]
        drug2_ddi_emb = ddi_d_embeds[drug2.long()]

        drug1_kg_emb = kg_d_embeds[drug1.long()]
        drug2_kg_emb = kg_d_embeds[drug2.long()]

        drug1_mole_emb = molecule_embeddings[drug1.long()]
        drug2_mole_emb = molecule_embeddings[drug2.long()]

        drug_kg_emb = torch.cat([drug1_kg_emb, drug2_kg_emb], axis=0)
        con_smile_emb = torch.cat([drug1_mole_emb, drug2_mole_emb], axis=0)

        kg_predict = DiffProcess.p_sample(KGDNet, drug_kg_emb, con_smile_emb, self.args.sampling_steps, self.args.sampling_noise)
        drug1_kg_demb, drug2_kg_demb = torch.chunk(kg_predict, 2, dim=0)

        
        drug1_emb = drug1_ddi_emb + drug1_kg_demb
        drug2_emb = drug2_ddi_emb + drug2_kg_demb


        pre_rating = self.sigmoid(torch.sum(drug1_emb * drug2_emb, 1))
        
        return pre_rating

    
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)
        
class RGAT(nn.Module):
    def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = agg_emb + entity_emb
        return agg_emb
        
    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
        return entity_res_emb



class KGDNet(nn.Module):


    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(KGDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + 768 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, con_smile_emb):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb, con_smile_emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):

    return tensor.mean(dim=list(range(1, len(tensor.shape))))