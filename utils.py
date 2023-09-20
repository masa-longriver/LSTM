import os
import torch
import datetime as dt
import matplotlib.pyplot as plt

class EarlyStopping():
    def __init__(self, config):
        self.config = config
        # 最良のモデルを保存する先
        self.path = os.path.join(os.getcwd(), "models")
        # lossの記録を保持する変数
        self.best_loss = float('inf')
        # 最小lossを更新できなかった連続回数
        self.patience = 0
    
    def check(self, loss, model):
        # 最小のlossを更新できた場合
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
          
            return False

        # lossを上回った場合
        else:
            self.patience += 1
            # patienceを上回った場合
            if self.patience >= self.config['model']['patience']:
                print("========== Early Stopping ===========")
                print(f"Best valid loss: {self.best_loss:.4f}")
                if not os.path.exists(self.path):
                    os.makedirs("models")
                
                file_nm = os.path.join(self.path, f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_LSTM")
                torch.save(model.state_dict(), file_nm)

                return True
            
            return False

# lossを保存
def save_loss(train_loss, valid_loss):
    path = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(path):
        os.makedirs("log")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title("losses")
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.legend()

    file_nm = os.path.join(path, f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_loss.png")
    plt.savefig(file_nm)
    plt.close()

# 予測結果を保存
def save_img(config, train_pred, valid_pred, test_pred, days, labels):
    window = config['data']['window']
    plt.figure(figsize=(15, 6))
    plt.plot(days, labels, label='Actual')
    plt.plot(days[window:len(train_pred)+window], train_pred, label='pred(train)')
    plt.plot(days[len(train_pred)+window: len(train_pred)+len(valid_pred)+window], valid_pred, label='pred(valid)')
    plt.plot(days[-len(test_pred):], test_pred, label='pred(test)')
    plt.legend()
    file_nm = os.path.join(os.getcwd(), 'log', f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_pred.png")
    plt.savefig(file_nm)
    plt.close()
