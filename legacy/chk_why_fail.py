import numpy as np

import util

SUCCESS_FILE = 'submission_7215344.csv'
FAIL_FILE = 'submission.csv'

with open(SUCCESS_FILE, 'r') as fin:
    cnt_su = fin.read().splitlines()
    header_su = cnt_su[0]
    chid_su = sorted([int(line.split(',')[0]) for line in cnt_su[1:]])

with open(FAIL_FILE, 'r') as fin:
    cnt_fa = fin.read().splitlines()
    header_fa = cnt_su[0]
    chid_fa = sorted([int(line.split(',')[0]) for line in cnt_fa[1:]])

assert header_su == header_fa

assert len(chid_su) == len(chid_fa)

for id_s, id_f in zip(chid_su, chid_fa):
    assert id_s == id_f

for line in cnt_fa[1:]:
    _, t1, t2, t3 = line.split(',')
    assert int(t1) in util.TARGET_TAGS
    assert int(t2) in util.TARGET_TAGS
    assert int(t3) in util.TARGET_TAGS
    assert len(set([t1, t2, t3])) == 3
