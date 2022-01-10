import numpy as np

import util

customer_to_records = {}
with open('data/tbrain_cc_training_48tags_hash_final.csv', 'r') as fin:
    line_counter = 0
    header = fin.readline().rstrip()
    # for i, h in enumerate(header.split(',')):
    #     print(i, h)
    while True:
        line = fin.readline().rstrip()
        if not line:
            break
        if line_counter % 100000 == 0:
            print('Parsing line', line_counter)
        line_counter += 1
        chid, rec = util.parse_line_to_mat(line)
        if chid not in customer_to_records:
            customer_to_records[chid] = []
        customer_to_records[chid].append(rec)
        # if line_counter == 5000000:
        #     break

print('Num lines:', line_counter)
print('Num customers:', len(customer_to_records))

print('Stacking array...')
for i, c in enumerate(customer_to_records):
    if i % 10000 == 0:
        print('\tStacking customer {}/{}...'.format(i, len(customer_to_records)))
    customer_to_records[c] = np.vstack(customer_to_records[c])

print('Writing...')
util.save_pkl('data/tbrain_cc_training_48tags_hash_final.pkl', customer_to_records)
