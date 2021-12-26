import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, datasplit_path, data_path, split='Train', to_fit=True, batch_size=4, shuffle=True, rotate=False):
        self.datasplit_path = datasplit_path
        self.data_path = data_path
        self.split = split
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rotate = rotate
        self.DATA_KEY = 'data'

        self._load_data()
        self.on_epoch_end()
    
    def _load_data(self):
        datasplit = pd.read_csv(self.datasplit_path)
        datasplit = datasplit.loc[datasplit['Split'] == self.split]
        dataframe = []
        for _, row in datasplit.iterrows():
            hdf5_path = os.path.join(self.data_path, row['Filename'])
            data = h5py.File(hdf5_path, 'r')
            dataframe.append(pd.DataFrame({'Filename':row['Filename'], 'index':range(len(data[self.DATA_KEY]))}))
        self.dataframe = pd.concat(dataframe)

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def _rotate(self, data):
        output = []
        for img in data:
            output.append(np.rot90(img, k=np.random.choice(4)))
        return np.array(output)

    def _get_data(self, index):
        X = []
        minid = self.batch_size * index
        maxid = self.batch_size * (index + 1)
        for _, row in self.dataframe.iloc[minid:maxid].iterrows():
            data = h5py.File(os.path.join(self.data_path, row['Filename']), 'r')
            X.append(data[self.DATA_KEY][row['index']])
        X = np.array(X)
        return X[:, :, :, np.newaxis]

    def __getitem__(self, index):
        X = self._get_data(index)
        Y = np.copy(X)
        if self.rotate:
            X = self._rotate(X)

        if self.to_fit:
            return X, Y
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)


if __name__ == '__main__':
    cdg = CustomDataGenerator(datasplit_path='Data/datasplit.csv',
        data_path='Data/hdf5/', batch_size=64, rotate=True)
    
    for idx in range(len(cdg)):
        x = cdg[idx]
        print("index: {:03d}".format(idx), x[0].shape)
