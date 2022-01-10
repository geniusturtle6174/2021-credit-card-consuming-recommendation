import os
import json
import time
import argparse

import torch
import numpy as np
from torch.autograd import Variable

import util
import model

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('modeldir', help='Model directory name')
parser.add_argument('model_postfix')
parser.add_argument('max_len', type=int)
args = parser.parse_args()

print('Loading network...')
save_dic = torch.load(os.path.join(args.modeldir, 'model{}'.format(args.model_postfix)))
network = model.BILSTM_MH_ATTN()
network.load_state_dict(save_dic)
network.eval()
network.to(device)

print('Loading pkl...')
tic = time.time()
customer_to_records = util.load_pkl('data/tbrain_cc_training_48tags_hash_final.pkl')
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

print('Reading csv data...')
with open('data/sample_submission.csv', 'r') as fin:
    cnt = fin.read().splitlines()
    submission_header = cnt[0]
    chid_to_eval = [int(line.split(',')[0]) for line in cnt[1:]]

tic = time.time()
with open('submission.csv', 'w') as fout:
    fout.write(submission_header + '\n')
    for i, c in enumerate(chid_to_eval):
        if i % 5000 == 0:
            print('Processing customer {}/{}...'.format(i, len(customer_to_records)))
            toc = time.time()
            print('\tTime elapsed (s):', toc - tic)
        fea_tag, fea_com = util.raw_mat_to_fea(customer_to_records[c], 'test', args.max_len)
        data_loader = torch.utils.data.DataLoader(
            util.Data2Torch({
                'fea': fea_tag[np.newaxis, :, :],
                'fea_com': fea_com[np.newaxis, :],
            }),
            batch_size=1,
        )
        for data in data_loader:
            pred = network(
                Variable(data['fea'].to(device)),
                Variable(data['fea_com'].to(device)),
            ).detach().cpu().numpy()[0]
        top_n_idx = np.argsort(-pred)[:3]
        fout.write('{},{},{},{}\n'.format(
            c,
            util.TARGET_TAGS[top_n_idx[0]],
            util.TARGET_TAGS[top_n_idx[1]],
            util.TARGET_TAGS[top_n_idx[2]],
        ))
