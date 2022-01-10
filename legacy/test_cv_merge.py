import numpy as np

import util

np.set_printoptions(linewidth=150)

print('Reading csv data...')
with open('data/sample_submission.csv', 'r') as fin:
    cnt = fin.read().splitlines()
    submission_header = cnt[0]
    chid_to_eval = [int(line.split(',')[0]) for line in cnt[1:]]

print('Loading npy files...')
result_0 = np.load('result_0.npy')
result_1 = np.load('result_1.npy')
result_2 = np.load('result_2.npy')
result_3 = np.load('result_3.npy')
result_4 = np.load('result_4.npy')
print('Finished, shape:', result_0.shape)

with open('submission.csv', 'w') as fout:
    fout.write(submission_header + '\n')
    for i, c in enumerate(chid_to_eval):
        if i % 5000 == 0:
            print('Processing customer {}/{}...'.format(i, len(chid_to_eval)))
        pred = np.vstack([
            result_0[i],
            result_1[i],
            result_2[i],
            result_3[i],
            result_4[i],
        ])
        pred = np.sum(pred, axis=0)
        top_n_idx = np.argsort(-pred)[:3]
        fout.write('{},{},{},{}\n'.format(
            c,
            util.TARGET_TAGS[top_n_idx[0]],
            util.TARGET_TAGS[top_n_idx[1]],
            util.TARGET_TAGS[top_n_idx[2]],
        ))
