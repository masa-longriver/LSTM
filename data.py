import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# 時系列データをwindowに分割
def devide_by_window(config, data, label):
    out_seq = []
    L = len(data)
    window = config['data']['window']
    for i in range(L - window):
        seq = torch.FloatTensor(data[i: i+window])
        seq_label = torch.FloatTensor(label[i: i+window])
        out_seq.append((seq, seq_label))
    
    return out_seq

# dataloaderを作成
def make_dataloader(config):
    config_data = config['data']
    df = pd.read_csv(config_data['path'])
    df = df.sort_values('日付').reset_index(drop=True)

    data = df[['始値', '高値', '安値', '終値']].values[:-1]
    label = df['終値'].values[1:]
    day = df['日付'].values[1:]

    seq = devide_by_window(config, data, label)

    # train, valid, testに分ける
    train_size = int(len(df) * config_data['train_size'])
    valid_size = int(len(df) * config_data['valid_size'])
    train_dataset = seq[:train_size]
    valid_dataset = seq[train_size: train_size+valid_size]
    test_dataset  = seq[train_size+valid_size: ]

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=config_data['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config_data['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config_data['batch_size'], shuffle=False)

    return train_loader, valid_loader, test_loader, day, label