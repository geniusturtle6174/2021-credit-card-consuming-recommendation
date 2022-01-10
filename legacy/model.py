import torch
import torch.nn.functional as F
from torch import nn


class BILSTM(nn.Module):
    def __init__(self):
        super(BILSTM, self).__init__()
        DIM_TAG = 491
        DIM_COM = 49
        self.pre = nn.Sequential(
            nn.Linear(in_features=DIM_TAG, out_features=32),
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.post = nn.Sequential(
            nn.Linear(in_features=32*2+DIM_COM, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
        )

    def forward(self, x_tag, x_com):
        h_tag = self.pre(x_tag)
        h_tag, _ = self.rnn(h_tag)
        h = torch.cat((h_tag[:, -1, :], x_com), dim=1)
        h = self.post(h)
        return h


class BILSTM_ATTN(nn.Module):
    def __init__(self):
        super(BILSTM_ATTN, self).__init__()
        DIM_TAG = 194
        DIM_COM = 49
        self.pre = nn.Sequential(
            nn.Linear(in_features=DIM_TAG, out_features=32),
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.attn_dense_h = nn.Linear(in_features=32, out_features=1)
        self.attn_dense_c = nn.Linear(in_features=32, out_features=1)
        self.attn_rnn = nn.LSTM(
            input_size=64,
            hidden_size=1,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.softmax = nn.Softmax(dim=1)
        self.post = nn.Sequential(
            nn.Linear(in_features=32*2+DIM_COM, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
        )

    def forward(self, x_tag, x_com):
        h_tag = self.pre(x_tag)
        h_tag, (hn, cn) = self.rnn(h_tag)
        hn = self.attn_dense_h(hn[-1])[None, :, :]
        cn = self.attn_dense_c(cn[-1])[None, :, :]
        attn, _ = self.attn_rnn(h_tag, (hn, cn))
        attn = self.softmax(attn)
        h_tag = attn * h_tag
        h_tag = torch.sum(h_tag, dim=1, keepdim=False)
        h = torch.cat((h_tag, x_com), dim=1)
        h = self.post(h)
        return h


class BILSTM_MH_ATTN(nn.Module):
    def __init__(self):
        super(BILSTM_MH_ATTN, self).__init__()
        DIM_TAG = 194
        DIM_COM = 20
        DIM_LSTM_HIDDEN = 32
        self.pre = nn.Sequential(
            nn.Linear(in_features=DIM_TAG, out_features=32),
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=DIM_LSTM_HIDDEN,
            num_layers=2,
            dropout=0.05,
            bidirectional=True,
            batch_first=True,
        )
        self.attn_dense_h_1 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_2 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_3 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_4 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_1 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_2 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_3 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_4 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_rnn_1 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_2 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_3 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_4 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)
        self.softmax_3 = nn.Softmax(dim=1)
        self.softmax_4 = nn.Softmax(dim=1)
        self.post = nn.Sequential(
            nn.Linear(in_features=DIM_LSTM_HIDDEN*2*4+DIM_COM, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
        )

    def forward(self, x_tag, x_com):
        # Pre + RNN
        h_tag = self.pre(x_tag)
        h_tag, (hn, cn) = self.rnn(h_tag)
        # ATTN
        hn_1 = self.attn_dense_h_1(hn[-1])[None, :, :]
        hn_2 = self.attn_dense_h_2(hn[-1])[None, :, :]
        hn_3 = self.attn_dense_h_3(hn[-2])[None, :, :]
        hn_4 = self.attn_dense_h_4(hn[-2])[None, :, :]
        cn_1 = self.attn_dense_c_1(cn[-1])[None, :, :]
        cn_2 = self.attn_dense_c_2(cn[-1])[None, :, :]
        cn_3 = self.attn_dense_c_3(cn[-2])[None, :, :]
        cn_4 = self.attn_dense_c_4(cn[-2])[None, :, :]
        attn_1, _ = self.attn_rnn_1(h_tag, (hn_1, cn_1))
        attn_2, _ = self.attn_rnn_2(h_tag, (hn_2, cn_2))
        attn_3, _ = self.attn_rnn_3(h_tag, (hn_3, cn_3))
        attn_4, _ = self.attn_rnn_4(h_tag, (hn_4, cn_4))
        attn_1 = self.softmax_1(attn_1)
        attn_2 = self.softmax_2(attn_2)
        attn_3 = self.softmax_3(attn_3)
        attn_4 = self.softmax_4(attn_4)
        h_tag_1 = attn_1 * h_tag
        h_tag_2 = attn_2 * h_tag
        h_tag_3 = attn_3 * h_tag
        h_tag_4 = attn_4 * h_tag
        h_tag_1 = torch.sum(h_tag_1, dim=1, keepdim=False)
        h_tag_2 = torch.sum(h_tag_2, dim=1, keepdim=False)
        h_tag_3 = torch.sum(h_tag_3, dim=1, keepdim=False)
        h_tag_4 = torch.sum(h_tag_4, dim=1, keepdim=False)
        # CONCAT + POST
        h = torch.cat((h_tag_1, h_tag_2, h_tag_3, h_tag_4, x_com), dim=1)
        h = self.post(h)
        # RETURN
        return h


class BILSTM_INIT_STATE_MH_ATTN(nn.Module):
    def __init__(self):
        super(BILSTM_INIT_STATE_MH_ATTN, self).__init__()
        DIM_TAG = 194
        DIM_COM = 49
        DIM_LSTM_HIDDEN = 32
        NUM_LSTM_LAYER = 2
        self.pre = nn.Linear(in_features=DIM_TAG, out_features=32)
        self.h_1_first = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.h_2_first = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.h_1_last = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.h_2_last = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.c_1_first = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.c_2_first = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.c_1_last = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.c_2_last = nn.Linear(in_features=DIM_COM, out_features=DIM_LSTM_HIDDEN)
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=DIM_LSTM_HIDDEN,
            num_layers=NUM_LSTM_LAYER,
            dropout=0.05,
            bidirectional=True,
            batch_first=True,
        )
        self.attn_dense_h_1 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_2 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_3 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_h_4 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_1 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_2 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_3 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_dense_c_4 = nn.Linear(in_features=DIM_LSTM_HIDDEN, out_features=1)
        self.attn_rnn_1 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_2 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_3 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.attn_rnn_4 = nn.LSTM(input_size=DIM_LSTM_HIDDEN*2, hidden_size=1, num_layers=1, batch_first=True)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)
        self.softmax_3 = nn.Softmax(dim=1)
        self.softmax_4 = nn.Softmax(dim=1)
        self.post = nn.Sequential(
            nn.Linear(in_features=DIM_LSTM_HIDDEN*2*4+DIM_COM*2, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=16),
            nn.Sigmoid(),
        )

    def forward(self, x_tag, x_com_first, x_com_last):
        # Initial states
        h_0 = torch.stack((
            self.h_1_first(x_com_first),
            self.h_1_last(x_com_last),
            self.h_2_first(x_com_first),
            self.h_2_last(x_com_last),
        ))
        c_0 = torch.stack((
            self.c_1_first(x_com_first),
            self.c_1_last(x_com_last),
            self.c_2_first(x_com_first),
            self.c_2_last(x_com_last),
        ))
        # Pre + RNN
        h_tag = self.pre(x_tag)
        h_tag, (hn, cn) = self.rnn(h_tag, (h_0, c_0))
        # ATTN
        hn_1 = self.attn_dense_h_1(hn[-1])[None, :, :]
        hn_2 = self.attn_dense_h_2(hn[-1])[None, :, :]
        hn_3 = self.attn_dense_h_3(hn[-2])[None, :, :]
        hn_4 = self.attn_dense_h_4(hn[-2])[None, :, :]
        cn_1 = self.attn_dense_c_1(cn[-1])[None, :, :]
        cn_2 = self.attn_dense_c_2(cn[-1])[None, :, :]
        cn_3 = self.attn_dense_c_3(cn[-2])[None, :, :]
        cn_4 = self.attn_dense_c_4(cn[-2])[None, :, :]
        attn_1, _ = self.attn_rnn_1(h_tag, (hn_1, cn_1))
        attn_2, _ = self.attn_rnn_2(h_tag, (hn_2, cn_2))
        attn_3, _ = self.attn_rnn_3(h_tag, (hn_3, cn_3))
        attn_4, _ = self.attn_rnn_4(h_tag, (hn_4, cn_4))
        attn_1 = self.softmax_1(attn_1)
        attn_2 = self.softmax_2(attn_2)
        attn_3 = self.softmax_3(attn_3)
        attn_4 = self.softmax_4(attn_4)
        h_tag_1 = attn_1 * h_tag
        h_tag_2 = attn_2 * h_tag
        h_tag_3 = attn_3 * h_tag
        h_tag_4 = attn_4 * h_tag
        h_tag_1 = torch.sum(h_tag_1, dim=1, keepdim=False)
        h_tag_2 = torch.sum(h_tag_2, dim=1, keepdim=False)
        h_tag_3 = torch.sum(h_tag_3, dim=1, keepdim=False)
        h_tag_4 = torch.sum(h_tag_4, dim=1, keepdim=False)
        # CONCAT + POST
        h = torch.cat((h_tag_1, h_tag_2, h_tag_3, h_tag_4, x_com_first, x_com_last), dim=1)
        h = self.post(h)
        # RETURN
        return h
