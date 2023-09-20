import json
import torch
import torch.nn as nn
import torch.optim as optim

from data import make_dataloader
from model import LSTM
from utils import EarlyStopping, save_loss, save_img
from run import run


if __name__ == '__main__':
    
    # configファイルの読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # seedの設定
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    # deviceの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    # 株価データの読み込み
    train_loader, valid_loader, test_loader, day, label = make_dataloader(config)

    # モデル、評価関数、活性化関数の定義
    model = LSTM(config).to(config['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['optim']['lr'])
    es = EarlyStopping(config)

    # 学習
    train_losses = []; valid_losses = []
    for epoch in range(config['epochs']):
        train_loss, _ = run(config, model, train_loader, criterion, optimizer, state='train')
        valid_loss, _ = run(config, model, valid_loader, criterion, optimizer, state='eval')
        if epoch % 1 == 0:
            print(f"Epoch: {epoch+1}  train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Early Stopping
        if es.check(valid_loss, model):
            break
    
    # 評価
    test_loss, test_pred = run(config, model, test_loader, criterion, optimizer, state='eval')
    print(f"test_loss: {test_loss:.4f}")

    # 画像出力するためにtrainとvalidの予測も入手する
    train_loss, train_pred = run(config, model, train_loader, criterion, optimizer, state='eval')
    valid_loss, valid_pred = run(config, model, valid_loader, criterion, optimizer, state='eval')

    # 画像出力
    save_loss(train_losses, valid_losses)
    save_img(config, train_pred, valid_pred, test_pred, day, label)