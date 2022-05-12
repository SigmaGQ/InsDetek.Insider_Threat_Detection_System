import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
from Model.evaluation import Evaluation
def train(model, epoch, dataset, optimizer, loss_function):
    epoch_bar = tqdm(range(epoch), leave = False)
    loss_list = []
    metric = []

    for epoch_i, epoch in enumerate(epoch_bar): 

        batch_bar = tqdm(dataset.train, leave = False)
        loss_list_epoch = []

        for batch_i, (feature_seqs, label_seqs, mask_seqs) in enumerate(batch_bar): # get feature sequenceS, label sequenceS and mask sequenceS
            
            # == Step 1. clear gradient ==
            model.zero_grad()
            # == Step 2. Run forward pass ==
            predict_seqs = model(feature_seqs)
            # == Step 3.1 Compute the loss ==
            seq_len = feature_seqs.shape[1] 
            predict_seqs = predict_seqs.reshape([-1,dataset.num_class,seq_len]) # [20,72,2]→[20,2,72] Because the input of NLLLoss is in (batchsize N,numclass C, d1, d2, ...)
            # predict_seqs=[bs N, 2, seq_len], label_seqs[bs N, seq_len]
            loss = loss_function(predict_seqs, label_seqs) # loss.shape = [batchsize, seq_len] = [20,72]
            loss = torch.mul(loss, mask_seqs).reshape(-1)
            loss = loss.sum() / mask_seqs.sum()
            # == Step 3.2 Compute the gradients ==
            loss.backward()
            # == Step 3.3 Update the parameters ==
            optimizer.step()
            loss_list_epoch.append(loss)
        
        # Evaluation
        val = Evaluation(model, dataset, loss_function)
        val('valid').get_metrics()
        metric.append(np.concatenate([[val.avg_loss],val.precision,val.recall,val.fscore]))
        loss_list.append(loss_list_epoch)

        torch.save({'model':model}, 'log/saved model/epoch{}.pth'.format(epoch_i))
        epoch_bar.set_description('Epoch: %i' % epoch)
        epoch_bar.set_postfix(valid_recall = '[0: {0:.3f}, 1: {1:.3f}]'.format(val.recall[0], val.recall[1])) 

    return loss_list, pd.DataFrame(metric, columns = ['avg_loss', 'precision_0', 'precision_1', 'recall_0', 'recall_1', 'f1_0', 'f1_1'])