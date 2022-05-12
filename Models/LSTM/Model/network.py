import torch
from torch import nn
import torch.nn.functional as F

class LSTM_network(nn.Module):

    def __init__(self, input_size, num_class, batch_size):
        super(LSTM_network, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_size, 256, kernel_size = 1) # [bs, f, seq] → [bs, 128, seq]
        self.conv2 = torch.nn.Conv1d(256, 128, kernel_size = 1) # [bs, f, seq] → [bs, 128, seq]
        self.lstm = nn.LSTM(128, 64, batch_first = True)
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 8)
        self.hidden2tag = nn.Linear(8, num_class)


    def forward(self, input):
        self.input_seq = input.permute(0,2,1) # to [batchsize, feature, seq]
        self.cnn_out1 = self.conv1(self.input_seq)        
        self.cnn_out2 = self.conv2(self.cnn_out1)
        self.cnn_out2 = self.cnn_out2.permute(0,2,1) # to [batchsize, seq, feature]
        self.lstm_out, (self.hidden, self.cell) = self.lstm(self.cnn_out2)
        hidden1_out = self.hidden1(self.lstm_out)
        hidden2_out = self.hidden2(hidden1_out)
        logit = self.hidden2tag(hidden2_out)
        log_prob = F.log_softmax(logit, dim=2)
        return log_prob