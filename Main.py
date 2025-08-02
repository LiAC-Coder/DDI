import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import *
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import random
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from models import diffusion_process as dp

class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.smiles_features = self.handler.smiles_features
    
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        log('Model Initialized')

        self.trainEpoch()
            

    def prepareModel(self):
        self.model = Model(self.handler)
        
        output_dims = [args.dims] + [args.latdim]
        input_dims = output_dims[::-1]

        self.KGDNet = KGDNet(input_dims, output_dims, args.latdim, time_type="cat", norm=args.norm).to(self.device)

        self.DiffProcess = dp.DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, self.device).to(self.device)
        
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.reg)
        self.optimizer2 = torch.optim.Adam(self.KGDNet.parameters(), lr=args.difflr, weight_decay=args.reg)
        self.rs_criterion = torch.nn.BCELoss()
        
    def trainEpoch(self):

        train_data = self.handler.train_data
        test_data  = self.handler.test_data
        eval_data  = self.handler.eval_data
        self.model.train()
        for current_epoch in range(args.epoch):
            dataloader = Data.DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle = True)
        
            epoch_loss = 0 
            for batch in dataloader:
                drug1, drug2, rating = batch[:,0].long(), batch[:,1].long(), batch[:,2].float()
                ddi_d_embeds, kg_d_embeds = self.model()

                drug1_mole_emb = self.smiles_features[drug1]
                drug2_mole_emb = self.smiles_features[drug2]

                drug1_kg_emb = kg_d_embeds[drug1]
                drug2_kg_emb = kg_d_embeds[drug2]

                drug_kg_emb = torch.cat([drug1_kg_emb, drug2_kg_emb], axis=0)
                con_smile_emb = torch.cat([drug1_mole_emb, drug2_mole_emb], axis=0)
                
                kg_terms = self.DiffProcess.caculate_losses(self.KGDNet, drug_kg_emb, con_smile_emb, args.reweight)
                kgelbo = kg_terms["loss"].mean()

                drug1_kg_demb, drug2_kg_demb = torch.chunk(kg_terms["pred_xstart"], 2, dim=0)

                
                drug1_ddi_emb = ddi_d_embeds[drug1]
                drug2_ddi_emb = ddi_d_embeds[drug2]
                cl_drug1 = self.contrastLoss(drug1_ddi_emb.detach(), drug1_kg_demb, args.temp) 
                cl_drug2 = self.contrastLoss(drug2_ddi_emb.detach(), drug2_kg_demb, args.temp) 
                conloss = args.factor * (cl_drug1 + cl_drug2)


                final_embeddings = (ddi_d_embeds, kg_d_embeds)
                pre_rating = self.model.get_rating(final_embeddings, drug1, drug2, (drug1_kg_demb, drug2_kg_demb)).float()
                loss = self.rs_criterion(pre_rating, rating.float()) + kgelbo + conloss

                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer2.step()

                batch_loss = loss.item()
                epoch_loss = epoch_loss + batch_loss + conloss

            self.test(current_epoch, train_data, test_data, eval_data, args, self.model)

    def contrastLoss(self, embeds1, embeds2, temp):
        pckEmbeds1 = F.normalize(embeds1, p=2)
        pckEmbeds2 = F.normalize(embeds2, p=2)
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
        return -torch.log(nume / deno).mean()
        
    def test(self, epoch, train_data, test_data, eval_data, args, ddi_model):

        ddi_model.eval()
        ddi_d_embeds, kg_d_embeds = ddi_model()
        final_embeddings = (ddi_d_embeds, kg_d_embeds)

        print('Epoch: ', epoch)

        tes_loader = DataLoader(dataset=test_data, batch_size = args.batch_size, shuffle = True)
        eva_loader = DataLoader(dataset=eval_data, batch_size = args.batch_size, shuffle = True)
        

        start = 0
        test_auc_list = []
        test_acc_list = []
        test_f1_list  = []
        test_aupr_list = []

        tqdm_tes_loader = tqdm(tes_loader)
        for iteration, batch in enumerate(tqdm_tes_loader, start=1):
            drug1, drug2, rating = batch[:,0].float(), batch[:,1].float(), batch[:,2].float()

            pre_rating = ddi_model.get_test_rating(final_embeddings, drug1, drug2, self.smiles_features, 
                                                      kg_d_embeds, self.DiffProcess, self.KGDNet).float()

            test_auc, test_acc, test_f1, test_aupr = auc_acc(rating.data.numpy(), pre_rating.data.numpy())

            test_auc_list.append(test_auc)
            test_acc_list.append(test_acc)
            test_f1_list.append(test_f1)
            test_aupr_list.append(test_aupr)

        log_test = 'Test set results:  '+'test_AUC: {:.4f}'.format(float(np.mean(test_auc_list)))+'  test_ACC: {:.4f}'.format(float(np.mean(test_acc_list)))+'  test_F1: {:.4f}'.format(float(np.mean(test_f1_list)))+'  test_aupr: {:.4f}'.format(float(np.mean(test_aupr_list)))
        print(log_test)


        start = 0
        eval_auc_list = []
        eval_acc_list = []
        eval_f1_list  = []
        eval_aupr_list = []


        tqdm_eva_loader = tqdm(eva_loader)
        for iteration, batch in enumerate(tqdm_eva_loader, start=1):
            drug1, drug2, rating = batch[:,0].float(), batch[:,1].float(), batch[:,2].float()
             
            pre_rating = ddi_model.get_test_rating(final_embeddings, drug1, drug2, self.smiles_features, 
                                                      kg_d_embeds, self.DiffProcess, self.KGDNet).float()

            eval_auc, eval_acc, eval_f1, eval_aupr = auc_acc(rating.data.numpy(), pre_rating.data.numpy())

            eval_auc_list.append(eval_auc)
            eval_acc_list.append(eval_acc)
            eval_f1_list.append(eval_f1)
            eval_aupr_list.append(eval_aupr)

        log_eval = 'Eval set results:  '+'eval_AUC: {:.4f}'.format(float(np.mean(eval_auc_list)))+'  eval_ACC: {:.4f}'.format(float(np.mean(eval_acc_list)))+'  eval_F1: {:.4f}'.format(float(np.mean(eval_f1_list)))+'  eval_aupr: {:.4f}'.format(float(np.mean(eval_aupr_list)))
        print(log_eval)
        


def auc_acc(labels, pred):
    auc = roc_auc_score(labels, pred)
    predictions = [1 if i >= 0.5 else 0 for i in pred]
    acc = np.mean(np.equal(predictions, labels))

    p, r, t = precision_recall_curve(y_true=labels.flatten(), probas_pred=pred.flatten())
    aupr = m.auc(r, p)
    
    scores = pred.copy()
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    
    f1 = f1_score(y_true=labels, y_pred=scores)

    return auc, acc, f1, aupr



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()

