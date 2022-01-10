import numpy as np

import util

CARD_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'other')

customer_to_dt_tag_records = {}
customer_to_fea_records = {}
card_used_counter = [0] * 15
card_cnt_counter = []
cat_all = {
    'masts': set(),
    'educd': set(),
    'trdtp': set(),
    'naty': set(),
    'poscd': set(),
    'cuorg': set(),
    'gender_code': set(),
    'age': set(),
    'primary_card': set(),
    # 'slam': set(),
}
tags_conut_counter = [0] * 50
tags_slam_counter = [0] * 50

with open('data/tbrain_cc_training_48tags_hash_final.csv', 'r') as fin:
    line_counter = 0
    header = fin.readline().rstrip()
    while True:
        line = fin.readline().rstrip()
        if not line:
            break
        if line_counter % 100000 == 0:
            print('Parsing line', line_counter)
        line_counter += 1
        chid, dt, shop_tag, rec = util.parse_line_to_dic(line)
        if chid not in customer_to_dt_tag_records:
            customer_to_dt_tag_records[chid] = {}
            customer_to_fea_records[chid] = {
                'masts': set(),
                'educd': set(),
                'trdtp': set(),
                'poscd': set(),
                'gender_code': set(),
                'age': set(),
                'primary_card': set(),
            }
        if dt not in customer_to_dt_tag_records[chid]:
            customer_to_dt_tag_records[chid][dt] = {}
        if shop_tag not in customer_to_dt_tag_records[chid][dt]:
            customer_to_dt_tag_records[chid][dt][shop_tag] = 0
        customer_to_dt_tag_records[chid][dt][shop_tag] += 1
        customer_to_fea_records[chid]['masts'].add(rec['masts'])
        customer_to_fea_records[chid]['educd'].add(rec['educd'])
        customer_to_fea_records[chid]['trdtp'].add(rec['trdtp'])
        customer_to_fea_records[chid]['poscd'].add(rec['poscd'])
        customer_to_fea_records[chid]['gender_code'].add(rec['gender_code'])
        customer_to_fea_records[chid]['age'].add(rec['age'])
        customer_to_fea_records[chid]['primary_card'].add(rec['primary_card'])
        for idx, cid in enumerate(CARD_IDS):
            card_used_counter[idx] += float(rec['card_{}_txn_cnt'.format(cid)]) > 0
        card_cnt_counter.append(len([None for cid in CARD_IDS if float(rec['card_{}_txn_cnt'.format(cid)]) > 0]))
        cat_all['masts'].add(rec['masts'])
        cat_all['educd'].add(rec['educd'])
        cat_all['trdtp'].add(rec['trdtp'])
        cat_all['naty'].add(rec['naty'])
        cat_all['poscd'].add(rec['poscd'])
        cat_all['cuorg'].add(rec['cuorg'])
        cat_all['gender_code'].add(rec['gender_code'])
        cat_all['age'].add(rec['age'])
        cat_all['primary_card'].add(rec['primary_card'])
        tags_conut_counter[shop_tag] += 1
        tags_slam_counter[shop_tag] += util.parse_amt(rec['txn_amt'])
        # cat_all['slam'].add(rec['slam'])
        # if line_counter == 100000:
        #     break

# print('\n'.join(header.split(',')))
print('Num lines:', line_counter)
print('Num customers:', len(customer_to_dt_tag_records))
print('----------')

num_dt_each_customer = [len(r) for _, r in customer_to_dt_tag_records.items()]
print('Stat of #dt: min {}, mean {}, median {}, max {}'.format(
    min(num_dt_each_customer),
    np.mean(num_dt_each_customer),
    np.median(num_dt_each_customer),
    max(num_dt_each_customer),
))
print('----------')

num_tag_one_dt_each_customer = [len(tag) for _, r in customer_to_dt_tag_records.items() for dt, tag in r.items()]
print('Stat of #tag in one dt: min {}, mean {:.4f}, median {}, q-75 {}, q-90 {}, q-91 {}, q-92 {}, q-93 {}, q-94 {}, q-95 {}, q-96 {}, q-97 {}, q-98 {}, q-99 {}, max {}'.format(
    min(num_tag_one_dt_each_customer),
    np.mean(num_tag_one_dt_each_customer),
    np.median(num_tag_one_dt_each_customer),
    np.quantile(num_tag_one_dt_each_customer, 0.75),
    np.quantile(num_tag_one_dt_each_customer, 0.90),
    np.quantile(num_tag_one_dt_each_customer, 0.91),
    np.quantile(num_tag_one_dt_each_customer, 0.92),
    np.quantile(num_tag_one_dt_each_customer, 0.93),
    np.quantile(num_tag_one_dt_each_customer, 0.94),
    np.quantile(num_tag_one_dt_each_customer, 0.95),
    np.quantile(num_tag_one_dt_each_customer, 0.96),
    np.quantile(num_tag_one_dt_each_customer, 0.97),
    np.quantile(num_tag_one_dt_each_customer, 0.98),
    np.quantile(num_tag_one_dt_each_customer, 0.99),
    max(num_tag_one_dt_each_customer),
))
print('----------')

num_dt_tag_each_customer = [c for _, r in customer_to_dt_tag_records.items() for dt, tag in r.items() for _, c in tag.items()]
print('Stat of #dt_tag: min {}, mean {}, median {}, max {}'.format(
    min(num_dt_tag_each_customer),
    np.mean(num_dt_tag_each_customer),
    np.median(num_dt_tag_each_customer),
    max(num_dt_tag_each_customer),
))
print('----------')

print('Times of card used:')
for i, c in enumerate(card_used_counter):
    print(i+1, c)
print('----------')

print('Stat of card_cnt_counter: min {}, mean {:.4f}, median {}, q-75 {}, q-90 {}, q-91 {}, q-92 {}, q-93 {}, q-94 {}, q-95 {}, q-96 {}, q-97 {}, q-98 {}, q-99 {}, max {}'.format(
    min(card_cnt_counter),
    np.mean(card_cnt_counter),
    np.median(card_cnt_counter),
    np.quantile(card_cnt_counter, 0.75),
    np.quantile(card_cnt_counter, 0.90),
    np.quantile(card_cnt_counter, 0.91),
    np.quantile(card_cnt_counter, 0.92),
    np.quantile(card_cnt_counter, 0.93),
    np.quantile(card_cnt_counter, 0.94),
    np.quantile(card_cnt_counter, 0.95),
    np.quantile(card_cnt_counter, 0.96),
    np.quantile(card_cnt_counter, 0.97),
    np.quantile(card_cnt_counter, 0.98),
    np.quantile(card_cnt_counter, 0.99),
    max(card_cnt_counter),
))
print('----------')

print('Counts and slam:')
for i in range(1, 50):
    print('{}\t{}\t{:.4f}'.format(i, tags_conut_counter[i], tags_slam_counter[i]))
print('----------')

print('Overall unique values:')
for cat_fea in sorted(list(cat_all.keys())):
    print(cat_fea, sorted(list(cat_all[cat_fea])))
print('----------')

print('Stat of unique values of each person:')
for key in ('masts', 'educd', 'trdtp', 'poscd', 'gender_code', 'age', 'primary_card'):
    raw_arr = [len(r[key]) for _, r in customer_to_fea_records.items()]
    is_missing = ['' in r[key] for _, r in customer_to_fea_records.items()]
    print('Stat of {} of each person: min {}, mean {}, median {}, max {}, missing {}'.format(
        key, min(raw_arr), np.mean(raw_arr), np.median(raw_arr), max(raw_arr), np.sum(is_missing)
    ))
