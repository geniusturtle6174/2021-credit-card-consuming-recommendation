import time

import numpy as np

import util

LEN_AT_LEAST = 15
MAX_NUM_FOR_ONE_CUSTOMER = 4

print('Loading pkl...')
tic = time.time()
customer_to_records = util.load_pkl('data/tbrain_cc_training_48tags_hash_final.pkl')
toc = time.time()
print('Loading finished, time elapsed (s):', toc - tic)

fea_tag_all = []
# fea_com_first_all = []
fea_com_last_all = []
ans_all = []
tic = time.time()
for i, c in enumerate(customer_to_records):
    if i % 10000 == 0:
        print('Processing customer {}/{}...'.format(i, len(customer_to_records)))
        toc = time.time()
        print('\tTime elapsed (s):', toc - tic)
    fea_tag, fea_com_last, ans = util.raw_mat_to_fea(
        customer_to_records[c], 'train', LEN_AT_LEAST, max_num_one_cus=MAX_NUM_FOR_ONE_CUSTOMER
    )
    if fea_tag is None:
        continue
    fea_tag_all.append(fea_tag)
    # fea_com_first_all.append(fea_com_first)
    fea_com_last_all.append(fea_com_last)
    ans_all.append(ans)
    # if i == 10000:
    #     break

del customer_to_records

print('Concatenating to one array...')
fea_tag_all = np.concatenate(fea_tag_all)
# fea_com_first_all = np.concatenate(fea_com_first_all)
fea_com_last_all = np.concatenate(fea_com_last_all)
ans_all = np.concatenate(ans_all)
print('Finished, shape:', fea_tag_all.shape, fea_com_last_all.shape, ans_all.shape)

np.save('fea_tag_all.npy', fea_tag_all)
# np.save('fea_com_first_all.npy', fea_com_first_all)
np.save('fea_com_last_all.npy', fea_com_last_all)
np.save('ans_all.npy', ans_all)
