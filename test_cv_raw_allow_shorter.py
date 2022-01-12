import argparse
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

import model_allow_shorter
import util

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('modeldir', help='Model directory name')
parser.add_argument('max_len', type=int)
parser.add_argument('--model_postfix', default='_bestVa')
parser.add_argument('--n_fold_train', type=int, default=5)
parser.add_argument('--test_batch_size', type=int, default=512)
args = parser.parse_args()

print('Loading network...')
networks = {}
for fold in range(args.n_fold_train):
    save_dic = torch.load(os.path.join(args.modeldir, 'model_{}{}'.format(fold, args.model_postfix)))
    networks[fold] = model_allow_shorter.BILSTM_MH_ATTN()
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

chid_to_fea = {}
for i, c in enumerate(chid_to_eval):
    if i % 10000 == 0:
        print('Feature extraction for {}/{}...'.format(i, len(chid_to_eval)))
    fea_tag, fea_com = util.raw_mat_to_fea(customer_to_records[c], 'test', args.max_len)
    chid_to_fea[c] = {
        'tag': fea_tag,
        'com': fea_com,
    }

tic = time.time()
for fold in range(args.n_fold_train):
    result_all = []
    chid_all = []
    fea_tag_batch = []
    fea_com_batch = []
    chid_batch = []
    for i, c in enumerate(chid_to_eval):
        if i % 5000 == 0:
            print('Fold {}: processing customer {}/{}...'.format(fold, i, len(customer_to_records)))
            toc = time.time()
            print('\tTime elapsed (s):', toc - tic)
        fea_tag_batch.append(chid_to_fea[c]['tag'])
        fea_com_batch.append(chid_to_fea[c]['com'])
        chid_batch.append(np.array(c))
        if len(fea_tag_batch) == args.test_batch_size:
            data_loader = torch.utils.data.DataLoader(
                util.Data2Torch({
                    'fea': fea_tag_batch,
                    'fea_com': fea_com_batch,
                    'chid': chid_batch,
                }),
                batch_size=args.test_batch_size,
                collate_fn=util.collate_fn_test,
            )
            for data, seq_len in data_loader:
                with torch.no_grad():
                    pred = networks[fold](
                        Variable(data['fea'].to(device)),
                        Variable(data['fea_com'].to(device)),
                        seq_len,
                    ).detach().cpu().numpy()
                result_all.append(pred)
                chid_all.append(data['chid'].numpy())
            fea_tag_batch = []
            fea_com_batch = []
            chid_batch = []
    if len(fea_tag_batch) > 0:
        data_loader = torch.utils.data.DataLoader(
            util.Data2Torch({
                'fea': fea_tag_batch,
                'fea_com': fea_com_batch,
                'chid': chid_batch,
            }),
            batch_size=len(fea_tag_batch),
            collate_fn=util.collate_fn_test,
        )
        for data, seq_len in data_loader:
            with torch.no_grad():
                pred = networks[fold](
                    Variable(data['fea'].to(device)),
                    Variable(data['fea_com'].to(device)),
                    seq_len,
                ).detach().cpu().numpy()
            result_all.append(pred)
            chid_all.append(data['chid'].numpy())
    result_all = np.vstack(result_all)
    chid_all = np.hstack(chid_all)
    print('Fold {} finished, shape: {}, {}'.format(fold, result_all.shape, chid_all.shape))
    np.save('result_{}.npy'.format(fold), result_all)
    np.save('chid_{}.npy'.format(fold), chid_all)
