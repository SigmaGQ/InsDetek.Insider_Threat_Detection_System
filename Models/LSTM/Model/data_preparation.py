import pandas as pd
from sklearn.model_selection import train_test_split
from Model.dataloader_ import df_to_tensor

class Data_preparation():
    def __init__(self, path, idx = None):
        """
        Args:
            path(str): path of csv
            idx(str): index_col
        Object attributes:
            path, idx, df: input dataframe
            user_list, user_train, user_valid, user_test: list of user#
            train_df, valid_df, test_df: dataframe of splited users
            batch_size: batchsize for dataloader
            train, valid tet: dataloader for train, valid and test
        """
        self.path = path
        self.idx = idx
        
    def read_data(self):
        self.df = pd.read_csv(self.path, index_col = self.idx)
        if 'week' in self.df.columns:
            self.df = self.df.drop('week', axis = 1)
        self.feat_size = len(self.df.columns)-2
        self.num_class = self.df['insider'].unique().size
        print("====== Read Data ======\nread '{0}', shape = {1}\n".format(self.path, self.df.shape))
        return self

    def split(self, size):
        """split data into train, valid, test set
        Args:
            df(DataFrame): input dataframe (must includes column 'user')
            size(list): [train_size, valid_size, user_size]
        """
        assert (sum(size) == 1) & (len(size) == 3), "input of 'size' should be three values with a sum of 1"

        self.user_list = self.df['user'].unique()
        self.user_train, self.user_test = train_test_split(self.user_list, train_size = size[0], shuffle = True)
        self.user_valid, self.user_test = train_test_split(self.user_test, train_size = size[1]/(1-size[0]), shuffle = True)

        self.train_df = self.df[self.df['user'].isin(self.user_train)]
        self.valid_df = self.df[self.df['user'].isin(self.user_valid)]
        self.test_df = self.df[self.df['user'].isin(self.user_test)]

        print('====== Split Data ======\nsize = ', size)
        print('train: {0} - {1} users\n'.format(self.train_df.shape, len(self.user_train)),
            '\rvalid: {0} - {1} users\n'.format(self.valid_df.shape, len(self.user_valid)),
            '\rtest : {0} - {1} users\n'.format(self.test_df.shape, len(self.user_test)))
        return self

    def dataloader(self, batch_size, all_label = True, print_summary = True, shuffle = True):
        """convert df to dataloader
        Args:
            all_df (tuple or list): train, valid and test data.
            all_label (bool): output will be the labels of the whole sequence if True,
                or the label of the last datapoint in the sequence otherwise.
            print_summary (bool, optional): print the size of output.
            shuffle (bool): parameter 'shuffle' in dataloader
        """
        self.out_df = []
        self.batch_size = batch_size

        print("====== DataLoader ======")
        if len(self.train_df) != 0:
            print("[{0} Data]".format('Train'), end=' ')
            self.train = df_to_tensor(self.train_df, batch_size, all_label, print_summary, shuffle)
        if len(self.valid_df) != 0:
            print("[{0} Data]".format('Valid'), end=' ')
            self.valid = df_to_tensor(self.valid_df, batch_size, all_label, print_summary, shuffle)
        if len(self.test_df) != 0:
            print("[{0} Data]".format('Test'), end=' ')
            self.test = df_to_tensor(self.test_df, batch_size, all_label, print_summary, shuffle)