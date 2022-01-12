import math
import pickle

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import yaml
from torch.utils.data import Dataset

TARGET_TAGS = np.array([2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48])
ONE_HOT_MAPS = {
    'masts': {k: v for v, k in enumerate([-1, 1, 2, 3])},
    'educd': {k: v for v, k in enumerate([-1] + list(range(1, 6+1)))},
    'trdtp': {k: v for v, k in enumerate([-1] + list(range(1, 29+1)))},
    'naty': {k: v for v, k in enumerate([-1, 1, 2])},
    'poscd': {k: v for v, k in enumerate([-1] + list(range(1, 10+1))+ [99])},
    'cuorg': {k: v for v, k in enumerate([
        -1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 35, 38, 40,
    ])},
    'gender_code': {k: v for v, k in enumerate([-1, 0, 1])},
    'age': {k: v for v, k in enumerate([-1] + list(range(1, 9+1)))},
    'primary_card': {k: v for v, k in enumerate([0, 1])},
}
FEA_NAME_TO_IDX = {
    'txn_cnt': 0,
    'txn_amt': 1,
    'domestic_offline_cnt': 2,
    'domestic_online_cnt': 3,
    'overseas_offline_cnt': 4,
    'overseas_online_cnt': 5,
    'domestic_offline_amt_pct': 6,
    'domestic_online_amt_pct': 7,
    'overseas_offline_amt_pct': 8,
    'overseas_online_amt_pct': 9,
    'card_1_txn_cnt': 10,
    'card_2_txn_cnt': 11,
    'card_3_txn_cnt': 12,
    'card_4_txn_cnt': 13,
    'card_5_txn_cnt': 14,
    'card_6_txn_cnt': 15,
    'card_7_txn_cnt': 16,
    'card_8_txn_cnt': 17,
    'card_9_txn_cnt': 18,
    'card_10_txn_cnt': 19,
    'card_11_txn_cnt': 20,
    'card_12_txn_cnt': 21,
    'card_13_txn_cnt': 22,
    'card_14_txn_cnt': 23,
    'card_other_txn_cnt': 24,
    'card_1_txn_amt_pct': 25,
    'card_2_txn_amt_pct': 26,
    'card_3_txn_amt_pct': 27,
    'card_4_txn_amt_pct': 28,
    'card_5_txn_amt_pct': 29,
    'card_6_txn_amt_pct': 30,
    'card_7_txn_amt_pct': 31,
    'card_8_txn_amt_pct': 32,
    'card_9_txn_amt_pct': 33,
    'card_10_txn_amt_pct': 34,
    'card_11_txn_amt_pct': 35,
    'card_12_txn_amt_pct': 36,
    'card_13_txn_amt_pct': 37,
    'card_14_txn_amt_pct': 38,
    'card_other_txn_amt_pct': 39,
    'masts': 40,
    'educd': 41,
    'trdtp': 42,
    'naty': 43,
    'poscd': 44,
    'cuorg': 45,
    'slam': 46,
    'gender_code': 47,
    'age': 48,
    'primary_card': 49,
}


def save_pkl(path, pkl):
    with open(path, 'wb') as handle:
        pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_cfg(path, cfg):
    with open(path, 'w') as fout:
        yaml.dump(cfg, fout, default_flow_style=False)


def print_and_write_file(fout, cnt, fout_end='\n'):
    print(cnt)
    if fout is not None:
        fout.write(cnt  + fout_end)
        fout.flush()


class Data2Torch(Dataset):
    def __init__(self, data):
        '''
        data: dict of np.array, should have key 'fea'
        '''
        self.data = {k: v for k, v in data.items()}

    def __getitem__(self, index):
        # for k, v in self.data.items():
        #     print(k, v[index].shape, v[index].dtype)
        # print('-----')
        return {k: torch.from_numpy(v[index]).float() for k, v in self.data.items()}

    def __len__(self):
        return len(self.data['fea'])


def collate_fn(data):
    data.sort(key=lambda x: x['fea'].shape[0], reverse=True)
    seq_lens = [d['fea'].shape[0] for d in data]
    data = {
        'fea': rnn_utils.pad_sequence([d['fea'] for d in data], batch_first=True),
        'fea_com_last': torch.stack([d['fea_com_last'] for d in data]),
        'ans': torch.stack([d['ans'] for d in data]),
    }
    return data, seq_lens


def collate_fn_test(data):
    data.sort(key=lambda x: x['fea'].shape[0], reverse=True)
    seq_lens = [d['fea'].shape[0] for d in data]
    data = {
        'fea': rnn_utils.pad_sequence([d['fea'] for d in data], batch_first=True),
        'fea_com': torch.stack([d['fea_com'] for d in data]),
        'chid': torch.stack([d['chid'] for d in data]),
    }
    return data, seq_lens


def one_dt_all_tags_to_ans(dt_data, ret_as_nparr=True):
    tag_amt = np.array([dt_data[tag][FEA_NAME_TO_IDX['txn_amt']] if tag in dt_data else 0 for tag in TARGET_TAGS])
    sort_tag_idx = np.argsort(-tag_amt)
    ret_ans = [0] * len(TARGET_TAGS)
    for i, sti in enumerate(sort_tag_idx):
        if tag_amt[sti] > 0: # Should buy something
            if i == 0: # Top 1
                ret_ans[sti] = 1
            elif i == 1: # Top 2
                ret_ans[sti] = 0.8
            elif i == 2: # Top 3
                ret_ans[sti] = 0.6
            else: # Buy something but not in top 3
                ret_ans[sti] = 0.2
    if ret_as_nparr:
        return np.array(ret_ans)
    return ret_ans


def get_one_hot_fea(raw_val, oh_map):
    '''
    idx: 1-based
    '''
    fea = [0] * len(oh_map)
    fea[oh_map[raw_val]] = 1
    return fea


def one_dt_one_tag_to_fea(dt_tag_data):
    if dt_tag_data is None:
        return [0] * 22

    return [
        dt_tag_data[FEA_NAME_TO_IDX['txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['txn_amt']],
        dt_tag_data[FEA_NAME_TO_IDX['domestic_offline_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['domestic_online_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['overseas_offline_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['overseas_online_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['domestic_offline_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['domestic_online_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['overseas_offline_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['overseas_online_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_1_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_2_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_4_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_6_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_10_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_other_txn_cnt']],
        dt_tag_data[FEA_NAME_TO_IDX['card_1_txn_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_2_txn_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_4_txn_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_6_txn_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_10_txn_amt_pct']],
        dt_tag_data[FEA_NAME_TO_IDX['card_other_txn_amt_pct']],
    ]


def one_dt_all_tags_to_tag_fea(dt_data, this_dt, base_dt, TOP_N=13):
    # Get top N tags
    tag_amt = np.array([dt_data[tag][FEA_NAME_TO_IDX['txn_amt']] if tag in dt_data else 0 for tag in range(1, 49+1)])
    top_n_tags = np.argsort(-tag_amt)[:TOP_N] + 1
    # Delta dt, DIM = 1
    ret_fea = [base_dt - this_dt]
    # Tag encoding, DIM = 49
    ret_fea += [0] * 49
    for n, tag in enumerate(top_n_tags):
        if tag_amt[tag-1] > 0:
            ret_fea[tag] = TOP_N - n # Change [0, 1, 2, ..., N-1] to [N, N-1, N-2, ..., 1]
    # Tag-related fea, DIM = 16 * N
    for tag in top_n_tags:
        if tag in dt_data:
            ret_fea += one_dt_one_tag_to_fea(dt_data[tag])
        else:
            ret_fea += one_dt_one_tag_to_fea(None)
    # # # Answer for this month, DIM = 16
    # ret_fea += one_dt_all_tags_to_ans(dt_data, ret_as_nparr=False)
    # Return
    return ret_fea, top_n_tags


def one_dt_all_tags_to_common_fea(dt_data, top_n_tags):
    ret_fea = []
    common_std = dt_data[top_n_tags[0]]
    ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['masts']], ONE_HOT_MAPS['masts']) # DIM = 4
    # ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['educd']], ONE_HOT_MAPS['educd']) # DIM = 7
    # ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['trdtp']], ONE_HOT_MAPS['trdtp']) # DIM = 30
    # ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['naty']], ONE_HOT_MAPS['naty']) # DIM = 3
    # ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['poscd']], ONE_HOT_MAPS['poscd']) # DIM = 12
    # ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['cuorg']], ONE_HOT_MAPS['cuorg']) # DIM = 35
    ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['gender_code']], ONE_HOT_MAPS['gender_code']) # DIM = 3
    ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['age']], ONE_HOT_MAPS['age']) # DIM = 10
    ret_fea += get_one_hot_fea(common_std[FEA_NAME_TO_IDX['primary_card']], ONE_HOT_MAPS['primary_card']) # DIM = 2
    ret_fea += [common_std[FEA_NAME_TO_IDX['slam']]]
    # Return
    return np.array(ret_fea)


def raw_mat_to_fea(mat, mode, len_at_least, max_num_one_cus=1, allow_shorter=False):
    # Aggregate by dt/tag
    dt_tag_all = {}
    for row in mat:
        dt = int(row[0])
        tag = int(row[1])
        if dt not in dt_tag_all:
            dt_tag_all[dt] = {}
        if tag not in dt_tag_all[dt]:
            dt_tag_all[dt][tag] = row[2:]
    dt_sorted = sorted(list(dt_tag_all.keys()))
    # Compose feature
    if mode == 'train':
        '''
        Return shape:
            fea_tag_all: (batch, time, fea)
            # fea_com_first_all: (batch, fea)
            fea_com_last_all: (batch, fea)
            ans_all: (batch, ans)
        '''
        if len(dt_sorted) <= (len_at_least + 1): # Shorter and equal
            if allow_shorter and len(dt_sorted) >= 2: # At least 1 for input and 1 for target
                fea_tag_one = []
                for dt in dt_sorted[:-1]:
                    fea_t, top_n_tags = one_dt_all_tags_to_tag_fea(dt_tag_all[dt], dt, dt_sorted[-1])
                    fea_tag_one.append(fea_t)
                fea_tag_one = np.stack(fea_tag_one).astype('float32')
                fea_com_last = one_dt_all_tags_to_common_fea(dt_tag_all[dt_sorted[-2]], top_n_tags).astype('float32')
                ans_one = one_dt_all_tags_to_ans(dt_tag_all[dt_sorted[-1]]).astype('float32')
                return fea_tag_one[np.newaxis, :, :], fea_com_last[np.newaxis, :], ans_one[np.newaxis, :]
            return None, None, None
        fea_tag_all = []
        # fea_com_first_all = []
        fea_com_last_all = []
        ans_all = []
        for look_back in range(max_num_one_cus):
            if len(dt_sorted) - len_at_least - look_back - 1 < 0:
                break
            fea_tag_one = []
            # fea_com_first_one = None
            for dt in dt_sorted[-(len_at_least+look_back+1):-(look_back+1)]:
                fea_t, top_n_tags = one_dt_all_tags_to_tag_fea(dt_tag_all[dt], dt, dt_sorted[-(look_back+1)])
                # if fea_com_first_one is None:
                #     fea_com_first_one = one_dt_all_tags_to_common_fea(dt_tag_all[dt], top_n_tags).astype('float32')
                fea_tag_one.append(fea_t)
            fea_tag_all.append(np.stack(fea_tag_one).astype('float32'))
            # fea_com_first_all.append(fea_com_first_one)
            fea_com_last_all.append(one_dt_all_tags_to_common_fea(dt_tag_all[dt_sorted[-(look_back+2)]], top_n_tags).astype('float32'))
            ans_all.append(one_dt_all_tags_to_ans(dt_tag_all[dt_sorted[-(look_back+1)]]).astype('float32'))
        fea_tag_all = np.stack(fea_tag_all).astype('float32')
        # fea_com_first_all = np.stack(fea_com_first_all).astype('float32')
        fea_com_last_all = np.stack(fea_com_last_all).astype('float32')
        ans_all = np.stack(ans_all).astype('float32')
        return fea_tag_all, fea_com_last_all, ans_all
    elif mode == 'test':
        '''
        Return shape:
            fea_tag_all: (time, fea)
            # fea_com_first_all: (fea)
            fea_com_last_all: (fea)
        '''
        fea_tag_all = []
        # fea_com_first_all = None
        for dt in dt_sorted[-len_at_least:]:
            fea_t, top_n_tags = one_dt_all_tags_to_tag_fea(dt_tag_all[dt], dt, 25)
            # if fea_com_first_all is None:
            #     fea_com_first_all = one_dt_all_tags_to_common_fea(dt_tag_all[dt], top_n_tags).astype('float32')
            fea_tag_all.append(fea_t)
        fea_tag_all = np.vstack(fea_tag_all).astype('float32')
        fea_com_last_all = one_dt_all_tags_to_common_fea(dt_tag_all[dt_sorted[-1]], top_n_tags).astype('float32')
        return fea_tag_all, fea_com_last_all
    else:
        raise Exception('Unsupported mode!')


def parse_float(in_str):
    if in_str == '':
        return -1
    return float(in_str)


def parse_amt(in_str):
    if in_str == '':
        return -1
    return math.log(float(in_str))


def parse_shop_tag(in_tag):
    if in_tag == 'other':
        return 49
    return int(in_tag)


def parse_line_to_mat(line):
    data = line.split(',')
    dt = int(data[0])
    chid = int(data[1])
    shop_tag = parse_shop_tag(data[2])
    return chid, np.array(
        [dt, shop_tag, parse_float(data[3]), parse_amt(data[4])] + \
        [parse_float(d) for d in data[5:49]] + \
        [parse_amt(data[49])] + \
        [parse_float(d) for d in data[50:]]
    ).astype('float32')


def parse_line_to_dic(line):
    data = line.split(',')
    dt = int(data[0])
    chid = int(data[1])
    shop_tag = parse_shop_tag(data[2])
    return chid, dt, shop_tag, {
        'txn_cnt': data[3],
        'txn_amt': data[4],
        'domestic_offline_cnt': data[5],
        'domestic_online_cnt': data[6],
        'overseas_offline_cnt': data[7],
        'overseas_online_cnt': data[8],
        'domestic_offline_amt_pct': data[9],
        'domestic_online_amt_pct': data[10],
        'overseas_offline_amt_pct': data[11],
        'overseas_online_amt_pct': data[12],
        'card_1_txn_cnt': data[13],
        'card_2_txn_cnt': data[14],
        'card_3_txn_cnt': data[15],
        'card_4_txn_cnt': data[16],
        'card_5_txn_cnt': data[17],
        'card_6_txn_cnt': data[18],
        'card_7_txn_cnt': data[19],
        'card_8_txn_cnt': data[20],
        'card_9_txn_cnt': data[21],
        'card_10_txn_cnt': data[22],
        'card_11_txn_cnt': data[23],
        'card_12_txn_cnt': data[24],
        'card_13_txn_cnt': data[25],
        'card_14_txn_cnt': data[26],
        'card_other_txn_cnt': data[27],
        'card_1_txn_amt_pct': data[28],
        'card_2_txn_amt_pct': data[29],
        'card_3_txn_amt_pct': data[30],
        'card_4_txn_amt_pct': data[31],
        'card_5_txn_amt_pct': data[32],
        'card_6_txn_amt_pct': data[33],
        'card_7_txn_amt_pct': data[34],
        'card_8_txn_amt_pct': data[35],
        'card_9_txn_amt_pct': data[36],
        'card_10_txn_amt_pct': data[37],
        'card_11_txn_amt_pct': data[38],
        'card_12_txn_amt_pct': data[39],
        'card_13_txn_amt_pct': data[40],
        'card_14_txn_amt_pct': data[41],
        'card_other_txn_amt_pct': data[42],
        'masts': data[43],
        'educd': data[44],
        'trdtp': data[45],
        'naty': data[46],
        'poscd': data[47],
        'cuorg': data[48],
        'slam': data[49],
        'gender_code': data[50],
        'age': data[51],
        'primary_card': data[52],
    }
