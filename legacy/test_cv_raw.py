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
parser.add_argument('max_len', type=int)
parser.add_argument('--model_postfix', default='_bestVa')
parser.add_argument('--n_fold_train', type=int, default=5)
args = parser.parse_args()

print('Loading network...')
networks = {}
for fold in range(args.n_fold_train):
    save_dic = torch.load(os.path.join(args.modeldir, 'model_{}{}'.format(fold, args.model_postfix)))
    networks[fold] = model.BILSTM_MH_ATTN()
    networks[fold].load_state_dict(save_dic)
    networks[fold].eval()
    networks[fold].to(device)

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
for fold in range(args.n_fold_train):
    result_all = []
    for i, c in enumerate(chid_to_eval):
        if i % 5000 == 0:
            print('Fold {}: processing customer {}/{}...'.format(fold, i, len(customer_to_records)))
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
            pred = networks[fold](
                Variable(data['fea'].to(device)),
                Variable(data['fea_com'].to(device)),
            ).detach().cpu().numpy()[0]
        result_all.append(pred)
    result_all = np.vstack(result_all)
    print('Fold {} finished, shape: {}'.format(fold, result_all.shape))
    np.save('result_{}.npy'.format(fold), result_all)
