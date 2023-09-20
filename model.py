import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config_model = config['model']
        self.input_dim  = self.config_model['input_dim']
        self.hidden_dim = self.config_model['hidden_dim']
        self.output_dim = self.config_model['output_dim']

        # 入力ゲート
        self.input_gate  = nn.Linear(self.input_dim+self.hidden_dim,
                                     self.hidden_dim)
        self.cell_gate   = nn.Linear(self.input_dim+self.hidden_dim,
                                     self.hidden_dim)
        # 忘却ゲート
        self.forget_gate = nn.Linear(self.input_dim+self.hidden_dim,
                                     self.hidden_dim)
        # 出力ゲート
        self.output_gate = nn.Linear(self.input_dim+self.hidden_dim,
                                     self.hidden_dim)
        
        # 出力層
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()
        self.dropout = nn.Dropout(self.config_model['dropout_rate'])
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 長期記憶と短期記憶の初期化
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)  # 短期記憶
        c = torch.zeros(batch_size, self.hidden_dim).to(x.device)  # 長期記憶

        # 各タイムステップでの計算
        for t in range(seq_len):
            combined = torch.cat((x[:,t,:], h), dim=1)
            combined = self.dropout(combined)
            i = self.sigmoid(self.input_gate(combined))   # 入力ゲートのsigmoid出力
            g = self.tanh(self.cell_gate(combined))       # 入力ゲートのtanh出力
            f = self.sigmoid(self.forget_gate(combined))  # 忘却ゲートのsigmoid出力
            o = self.sigmoid(self.output_gate(combined))  # 出力ゲートのsigmoid出力

            c = f * c + i * g     # 長期記憶の更新
            h = o * self.tanh(c)  # 短期記憶の更新
        
        h = self.dropout(h)
        out = self.output_layer(h)

        return out