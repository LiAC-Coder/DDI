import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.inf)

class DataHandler:
    def __init__(self):
        self.file_path = '../datasets/' + args.dataset + '/'

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def readTriplets(self, kg_matrix_extended):
        can_triplets_np = np.unique(kg_matrix_extended, axis=0)

        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_relations = max(triplets[:, 1]) + 1

        args.relation_num = n_relations

        args.entity_n = max(max(triplets[:, 0]), max(triplets[:, 1])) + 1

        return triplets
    
    def buildGraphs(self, triplets):

        kg_dict = defaultdict(list)
        kg_edges = list()


        kg_counter_dict = {}

        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            if h_id not in kg_counter_dict.keys():
                kg_counter_dict[h_id] = set()
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))


        return kg_edges, kg_dict
    
    def buildKGMatrix(self, kg_edges): 
        edge_list = []
        for h_id, t_id, r_id in kg_edges:
            edge_list.append((h_id, t_id))
        edge_list = np.array(edge_list)

        kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(args.entity_n, args.entity_n))

        return kgMatrix

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    
    def makeTorchAdj(self, mat):
        raw_mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(raw_mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape), raw_mat


    def buildDDIMatrix(self, mat):  
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape)

    def dataset_split(self, rating_np):

        eval_ratio = 0.2
        test_ratio = 0.2
        n_ratings = rating_np.shape[0]

        eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        train_data = rating_np[train_indices]
        eval_data = rating_np[eval_indices]
        test_data = rating_np[test_indices]

        return train_data, eval_data, test_data

    def LoadData(self):

        self.ddi_data = np.loadtxt(self.file_path + "ddi.txt", dtype=np.int32)
        self.num_drug = len(set(self.ddi_data[:, 0])|set(self.ddi_data[:, 1]))
        self.train_data, self.eval_data, self.test_data = self.dataset_split(self.ddi_data)
        args.num_drug = self.num_drug

        r_train = self.train_data[self.train_data[:, 2] == 1]
        train_csr = csr_matrix((np.ones(len(r_train)), (r_train[:, 0], r_train[:, 1])), shape=(self.num_drug, self.num_drug))
        self.trnMat = coo_matrix(train_csr)
        self.ddi_matrix = self.buildDDIMatrix(self.trnMat)
        self.torchDDIAdj, _ = self.makeTorchAdj(self.trnMat)

        KG_np = np.loadtxt(self.file_path + "kg.txt", dtype=np.int32)
        KG_np[:, [1, 2]] = KG_np[:, [2, 1]]
        kg_triplets = self.readTriplets(KG_np) 

        self.kg_edges, self.kg_dict = self.buildGraphs(kg_triplets)
        self.kg_matrix = self.buildKGMatrix(self.kg_edges)


        self.smiles_features = torch.from_numpy(np.load(self.file_path + "smiles_array.npy"))

    
class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    
    def __len__(self):
        return len(self.data)