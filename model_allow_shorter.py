import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn


class BILSTM_MH_ATTN(nn.Module):
    def __init__(self):
        super(BILSTM_MH_ATTN, self).__init__()
        DIM_TAG = 284
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

    def forward(self, x_tag, x_com, seq_len):
        # Pre + RNN
        h_tag = self.pre(x_tag)
        h_tag = rnn_utils.pack_padded_sequence(h_tag, lengths=seq_len, batch_first=True)
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
        pad_attn_1, _ = rnn_utils.pad_packed_sequence(attn_1, batch_first=True)
        pad_attn_2, _ = rnn_utils.pad_packed_sequence(attn_2, batch_first=True)
        pad_attn_3, _ = rnn_utils.pad_packed_sequence(attn_3, batch_first=True)
        pad_attn_4, _ = rnn_utils.pad_packed_sequence(attn_4, batch_first=True)
        pad_htag, _ = rnn_utils.pad_packed_sequence(h_tag, batch_first=True)
        attn_mask = pad_attn_1 ** 2
        attn_mask[attn_mask <= 0] = float('-inf')
        attn_mask[attn_mask > 0] = 0
        pad_attn_1 = self.softmax_1(pad_attn_1 + attn_mask)
        pad_attn_2 = self.softmax_2(pad_attn_2 + attn_mask)
        pad_attn_3 = self.softmax_3(pad_attn_3 + attn_mask)
        pad_attn_4 = self.softmax_4(pad_attn_4 + attn_mask)
        h_tag_1 = pad_attn_1 * pad_htag
        h_tag_2 = pad_attn_2 * pad_htag
        h_tag_3 = pad_attn_3 * pad_htag
        h_tag_4 = pad_attn_4 * pad_htag
        h_tag_1 = torch.sum(h_tag_1, dim=1, keepdim=False)
        h_tag_2 = torch.sum(h_tag_2, dim=1, keepdim=False)
        h_tag_3 = torch.sum(h_tag_3, dim=1, keepdim=False)
        h_tag_4 = torch.sum(h_tag_4, dim=1, keepdim=False)
        # CONCAT + POST
        h = torch.cat((h_tag_1, h_tag_2, h_tag_3, h_tag_4, x_com), dim=1)
        h = self.post(h)
        # RETURN
        return h
