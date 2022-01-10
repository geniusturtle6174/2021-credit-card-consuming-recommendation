import argparse

import numpy as np

import util

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('n_fold_train', type=int)
args = parser.parse_args()

print('Reading csv data...')
with open('data/sample_submission.csv', 'r') as fin:
    submission_header = fin.readline().rstrip()

print('Loading npy files...')
chid = []
result = []
for f in range(args.n_fold_train):
    chid.append(np.load('chid_{}.npy'.format(f)))
    result.append(np.load('result_{}.npy'.format(f)))
print('Finished, shape:', result[0].shape, chid[0].shape)

with open('submission.csv', 'w') as fout:
    fout.write(submission_header + '\n')
    for i in range(chid[0].shape[0]):
        if i % 5000 == 0:
            print('Processing customer {}/{}...'.format(i, chid[0].shape[0]))
        this_chid = chid[0][i]
        for f in range(1, args.n_fold_train):
            assert this_chid == chid[f][i]
        pred = np.vstack([result[f][i] for f in range(args.n_fold_train)])
        pred = np.sum(pred, axis=0)
        top_n_idx = np.argsort(-pred)[:3]
        fout.write('{:.0f},{},{},{}\n'.format(
            this_chid,
            util.TARGET_TAGS[top_n_idx[0]],
            util.TARGET_TAGS[top_n_idx[1]],
            util.TARGET_TAGS[top_n_idx[2]],
        ))
