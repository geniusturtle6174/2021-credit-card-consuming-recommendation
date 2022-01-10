import time
import argparse

import numpy as np

import util

parser = argparse.ArgumentParser()
parser.add_argument('--len_at_least', '-len', type=int, default=20)
parser.add_argument('--max_num_for_one_customer', '-max', type=int, default=4)
args = parser.parse_args()

print('Loading pkl...')
tic = time.time()
customer_to_records = util.load_pkl('data/tbrain_cc_training_48tags_hash_final.pkl')
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

fea_and_ans = {}
tic = time.time()
for i, c in enumerate(customer_to_records):
    if i % 10000 == 0:
        print('Processing customer {}/{}...'.format(i, len(customer_to_records)))
        toc = time.time()
        print('\tTime elapsed (s):', toc - tic)
    fea_tag, fea_com_last, ans = util.raw_mat_to_fea(
        customer_to_records[c],
        'train',
        args.len_at_least,
        max_num_one_cus=args.max_num_for_one_customer,
        allow_shorter=True
    )
    if fea_tag is None:
        continue
    key = fea_tag.shape[1]
    if key not in fea_and_ans:
        fea_and_ans[key] = {
            'fea_tag_all': [],
            'fea_com_last_all': [],
            'ans_all': [],
        }
    fea_and_ans[key]['fea_tag_all'].append(fea_tag)
    fea_and_ans[key]['fea_com_last_all'].append(fea_com_last)
    fea_and_ans[key]['ans_all'].append(ans)
    # if i == 10000:
    #     break

del customer_to_records

for key in fea_and_ans:
    print('Concatenating to one array for key {}...'.format(key))
    fea_tag_all = np.concatenate(fea_and_ans[key]['fea_tag_all'])
    fea_com_last_all = np.concatenate(fea_and_ans[key]['fea_com_last_all'])
    ans_all = np.concatenate(fea_and_ans[key]['ans_all'])
    print('\tFinished, shape:', fea_tag_all.shape, fea_com_last_all.shape, ans_all.shape)

    np.save('fea_tag_all_{}.npy'.format(key), fea_tag_all)
    np.save('fea_com_last_all_{}.npy'.format(key), fea_com_last_all)
    np.save('ans_all_{}.npy'.format(key), ans_all)
