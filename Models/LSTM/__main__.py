# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re

import torch
import torch.nn. as F
from sklearn.metrics import *

from Model.data_preparation import Data_preparation
from Model.network import LSTM_network
from Model.evaluation import Evaluation
from Model.trainer import train
from Model.focalloss import FocalLoss

"""# Read Data"""

data_path = 'data/data_3.1.csv'
split_size = [0.8, 0.1, 0.1]
batchsize = 32
data = Data_preparation(data_path, 'idx')
# data = data_preparation(data_path)
data.read_data().split(split_size).dataloader(batchsize)
# clear_output()

"""# Training"""

model = LSTM_network(input_size = data.feat_size, num_class = data.num_class, batch_size = data.batch_size)
loss_function = FocalLoss(gamma=5., alpha=0.85)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs_num = 200
loss_list, metric_df = train(model, epochs_num, data, optimizer, loss_function)
# loss_list_sf, valid_loss_list_sf, recall_list_sf = train(model, epochs_num, train_data, optimizer, loss_function)

"""# Results"""

loss_df = pd.DataFrame(columns = ['epoch', 'batch', 'loss'])
for i, epoch_i_loss in enumerate(loss_list):
    epoch_loss = []
    for batch_j_loss in epoch_i_loss:
        epoch_loss.append(float(batch_j_loss.detach().numpy()))
    df_temp = pd.DataFrame(columns = ['epoch', 'batch', 'loss'])
    df_temp['batch'] = list(range(len(epoch_i_loss)))
    df_temp['epoch'] = float(i)    
    df_temp['epoch'] = df_temp['epoch'].astype(float)
    df_temp['loss'] = epoch_loss
    loss_df = loss_df.append(df_temp)
loss_df = loss_df.reset_index(drop=True)

color_list = ['rgb({0}, {0}, {0})'.format(int(i/epochs_num*255)) for i in range(epochs_num)]
# fig = px.line(loss_df[loss_df['epoch']%10 == 0], x = 'batch', y = 'loss', color = 'epoch', color_discrete_sequence= color_list)
fig = px.line(loss_df, x = 'batch', y = 'loss', color = 'epoch', color_discrete_sequence= color_list)
fig.update_layout(plot_bgcolor='#a1afc9')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# #@title Train & Valid Loss

train_valid_loss = loss_df.groupby('epoch').mean().reset_index()
train_valid_loss['valid'] = 0

valid_loss = pd.DataFrame(columns = ['loss', 'valid'])
valid_loss['loss'] = metric_df['avg_loss']
valid_loss['loss'] = valid_loss['loss'].astype(float)
valid_loss['valid'] = 1
valid_loss = valid_loss.reset_index()
valid_loss = valid_loss.rename(columns={'index':'epoch'})
train_valid_loss = train_valid_loss.append(valid_loss)
px.line(train_valid_loss, x = 'epoch', y = 'loss', color = 'valid')

# @title Validation Recall
recall_df = metric_df[['recall_0', 'recall_1', 'precision_0', 'precision_1']].melt().reset_index().rename(columns={'recall_0':'recall of 0 (clean)', 'recall_1':'recall of 1(malicious)'})
recall_df['index'] = recall_df['index'].map(lambda x: x % epochs_num)
recall_df = recall_df.rename(columns={'index':'epoch', 'variable':'label', 'value':'rate'})
px.line(recall_df, x = 'epoch', y = 'rate', color = 'label')

#@title Recall by epochs
recall_df = metric_df[['recall_0','recall_1']].reset_index()
recall_df['index'] = recall_df['index'].astype(str)
fig = px.scatter(recall_df, x='recall_1', y='recall_0', color = 'index', color_discrete_sequence = color_list)
fig.update_layout(plot_bgcolor='#a1afc9')
fig.update_layout(title="Recall 0 vs 1", title_font_size=20)

#@title Precision-Recall by epochs
recall_df = metric_df[['precision_1','recall_1']].reset_index()
recall_df['index'] = recall_df['index'].astype(str)
fig = px.scatter(recall_df, x='precision_1', y='recall_1', color = 'index', color_discrete_sequence = color_list)
fig.update_layout(plot_bgcolor='#a1afc9')
fig.update_layout(title="P-R on 1", title_font_size=20)

#@title Model Performance
max_rate = [0, 0, 0, 0] # acc, tpr, fpr, mean
max_epoch = [0, 0, 0, 0]
rate_list = []
for i in range(200):
    best_model = torch.load('log/saved model/epoch'+str(i)+'.pth')['model']
    test = Evaluation(best_model, data)
    test('test').get_metrics()
    rate = [test.acc, test.tpr, test.fpr]
    rate_list.append(rate)

    for j in range(3):
        if rate[j] > max_rate[j]:
            max_rate[j] = rate[j]
            max_epoch[j] = i
    if np.mean(rate) > max_rate[3]:
        max_rate[3] = np.mean(rate)
        max_epoch[3] = i

print(max_epoch)
print(max_rate)
rate_df = pd.DataFrame(rate_list, columns = ['acc','tpr','fpr']).reset_index().melt(id_vars='index').rename(columns={'index':'epoch','variable':'metric','value':'rate'})
px.line(rate_df, x='epoch',y='rate',color='metric')

best_model = torch.load('log/saved model/epoch111.pth')['model']
test = Evaluation(best_model, data)
test('test').get_metrics(True)
print('   ACC     TPR     FPR')
print(format(test.acc,'.2%'),'|', format(test.tpr,'.2%'), '|', format(test.fpr,'.2%'))

"""# Save log"""

def write_log(exp_n, comment):
    exp = str(exp_n).zfill(3)
    path = 'log/exp' + exp + '/'
    if os.path.exists(path):
        print("=== Overwriting!!! ===")
    else:
        os.makedirs(path)   

    with open("log/log.txt","a") as f:
        f.write('\r\n\r\n' + exp)
        f.write('\r\n\t' + comment)

    dic = {'model':model, 'optim':optimizer}
    torch.save(dic, path +'model_optim.pth')

    print(path)
    return path

info =  '==== model info ====\n'\
        + 'batchsize=' + str(batch_size)\
        + '\n' + 'optim=' + optimizer.__class__.__name__\
        + '(lr={:g})'.format(optimizer.param_groups[0]['lr'])
print(info)
comment = """
data3.1.csv
100 epochs lr=0.0001, weight[1.5,5], Conv(256,ker=1)-Conv(128,ker=1)-LSTM64-Lin32-Lin8-Lin2
""" + info

exp_nums = re.findall('\d{3}', str(os.listdir('log')))
exp_new = max(list(map(lambda x: int(x), exp_nums))) + 1

path = write_log(exp_new, comment)

def loss_plot(loss_list, save=True):
    losslist = []
    for i in loss_list:
        losslist.append(float(i.detach().numpy()))
    note = '\n $\mathbf{data3.2, cov-cov-lstm-}$'
    plt.figure(figsize = (10,5))
    plt.suptitle(note + re.findall('LSTMTagger\(\\n(.+)\\n\)$',str(model), re.S)[0], y = -0.001)
    plt.subplot(1,2,1)
    plt.plot(losslist)
    plt.subplot(1,2,2)
    plt.plot(losslist)
    plt.ylim(0,5)
    if save:
        plt.savefig(path + 'loss.jpg', bbox_inches='tight')
    return losslist
# losslist = loss_plot(loss_list)

loss_df = pd.DataFrame(losslist)#.reset_index(drop=False)
loss_df.columns = ['loss']#['epoch', 'loss']
loss_df.to_csv(path+'loss.csv', index = False)