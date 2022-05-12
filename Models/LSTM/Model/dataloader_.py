
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from collections import defaultdict
import pandas as pd

class MyData(Dataset): 
        def __init__(self, x, y):
            self.feature = x
            self.label = y
        def __len__(self): 
            return len(self.feature)
        def __getitem__(self, idx):
            return (self.feature[idx], self.label[idx])
            
# sequences by user: df to tensor
def df_to_tensor(df, batch_size, all_label, print_summary = False, shuffle = False):
    """
    Args:
        df (DataFrame): sequences(both feature and label) in a DataFrame.
        all_label (bool): output will be the labels of the whole sequence if True,
                          or the label of the last datapoint in the sequence otherwise.
        print_summary (bool, optional): print the size of output.
        shuffle (bool): 'shuffle' parameter in DataLoader function
    """

    def collate_fn(feature_label): 

        # feature_label - List:[batchsize Tuples:(2 Tensors:[maxlen, dimension])] batchsize*2*maxlen*dimension

        features, labels = [], []
        # labels = torch.zeros(len(feature_label), len(feature_label[1][0]))
        for unit in feature_label:
            features.append(unit[0])
            labels.append(unit[1])

        # pad features
        seqs_len = [sequence.size(0) for sequence in features]
        padded_f = pad_sequence(features, batch_first=True, padding_value=0) # all seqs: fill zero to same length, and stack together
        

        # pad features labels
        seqs_len = [sequence.size(0) for sequence in labels]
        padded_l = pad_sequence(labels, batch_first=True, padding_value=0) # all seqs: fill zero to same length, and stack together

        # masks
        masks = [torch.ones(seq_len) for seq_len in seqs_len]
        padded_m = pad_sequence(masks, batch_first=True, padding_value=0)

        return (padded_f, padded_l, padded_m)
    from collections import defaultdict
    # dic of sequences in df format
    fea_df_dic = defaultdict(pd.DataFrame) # feature seq
    label_df = defaultdict(pd.DataFrame)# label seq
    for user in df.user.unique():
        fea_df_dic[user] = df[df.user == user].iloc[:, 1:-1]
        if all_label:
            # labels are the insider status of all activities in the sequence. [batchsize, maxlength (, 1)]
            label_df[user] = df[df.user == user].iloc[:, -1]
        else:
            # label is the insider status of the last activity in the sequence. [batchsize (, 1, 1)]
            label_df[user] = df[df.user == user].iloc[:, -1].iloc[[-1]]

    # list of features in tensor format (Need to be float32!)
    features = []
    for user, df_i in fea_df_dic.items():
        input_sub = torch.tensor(df_i.values).to(torch.float32)
        features.append(input_sub)
    # list of labels in tensor format (Need to be long!)
    labels = []
    for user, df_i in label_df.items():
        input_sub = torch.tensor(df_i.values).to(torch.long)
        labels.append(input_sub)

    # sequences tensors tuple to Dataset/DataLoader object
    data = MyData(features, labels)
    data_loader = DataLoader(data, batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last = True) 

    if print_summary:
        # print("==== DataFrame to DataLoader ====")
        print("Input DataFrame: {0} with {1:.2%} insiders".format(df.shape, df[df['insider'] != 0]['insider'].count()/df.shape[0]))
        print("   => {0} sequences (users)".format(len(features)))
        print("       features: {0}, labels: {1} in size of [(max) length, dimension]".format(list(features[0].shape), list(labels[0].shape)))
        print("   => {} batches in Dataloader".format(len(data_loader)), "(batchsize = {})".format(batch_size))
        print("       features: {0}, labels: {1}, masks: {2} in size of [batchsize, (max) length, dimension]\n".format(
            list(iter(data_loader).next()[0].shape), list(iter(data_loader).next()[1].shape), list(iter(data_loader).next()[2].shape)))
    return data_loader