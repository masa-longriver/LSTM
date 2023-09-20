import torch

def run(config, model, dataloader, criterion, optim, state='train'):
    running_loss = 0.0
    pred_list = []
    if state == 'train':
        model.train()                                   # 訓練モード
        for x, answer in dataloader:
            x = x.to(config['device'])                     # データをGPUへ
            optim.zero_grad()                           # パラメータ初期化
            pred = model(x)                          # モデル出力
            pred_list += list(pred)
            loss = criterion(pred, answer)                   # 評価関数に入れてlossを計算
            loss.backward()                             # 逆伝播
            optim.step()                                # パラメータ更新
            running_loss += loss.item() * x.size(0)
    
    elif state == 'eval':
        model.eval()                                    # 評価モードへ
        with torch.no_grad():                           # パラメータ更新しない
            for x, answer in dataloader:
                x = x.to(config['device'])
                pred = model(x)
                pred_list += list(pred)
                loss = criterion(pred, answer)
                running_loss += loss.item() * x.size(0)
    
    else:
        raise Exception("Choose from ['train', 'eval]")
    
    return running_loss / len(dataloader), pred_list
