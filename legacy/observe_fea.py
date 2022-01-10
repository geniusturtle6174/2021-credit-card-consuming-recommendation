import numpy as np
import matplotlib.pyplot as plt

fea_tag_all = np.load('fea_tag_all.npy')
fea_com_all = np.load('fea_com_last_all.npy')
ans_all = np.load('ans_all.npy')

print('Shapes:', fea_tag_all.shape, fea_com_all.shape, ans_all.shape)

print('NaN fea_tag_all:', np.sum(np.isnan(fea_tag_all)))
print('NaN fea_com_all:', np.sum(np.isnan(fea_com_all)))
print('NaN ans_all:', np.sum(np.isnan(ans_all)))

print('Inf fea_tag_all:', np.sum(np.isinf(fea_tag_all)))
print('Inf fea_com_all:', np.sum(np.isinf(fea_com_all)))
print('Inf ans_all:', np.sum(np.isinf(ans_all)))

print('Stat fea_tag_all:', np.max(fea_tag_all), np.min(fea_tag_all))
print('Stat fea_com_all:', np.max(fea_com_all), np.min(fea_com_all))

print('Stat of ans: min {}, mean {}, median {}, max {}'.format(
    np.min(ans_all),
    np.mean(ans_all),
    np.median(ans_all),
    np.max(ans_all),
))
for q in (0.840, 0.841, 0.842, 0.843, 0.844, 0.845, 0.846, 0.847, 0.848, 0.849, 0.850, 0.90):
	print('q-{}: {}'.format(q, np.quantile(ans_all, q)))

ans_sum = np.sum(ans_all, axis=1)
print('ans_sum.shape:', ans_sum.shape)
print('Stat of ans_sum: min {}, mean {}, median {}, max {}'.format(
    np.min(ans_sum),
    np.mean(ans_sum),
    np.median(ans_sum),
    np.max(ans_sum),
))
print('ans_sum zero num:', np.sum(ans_sum==0))

# observe_idx = 1999

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(fea_tag_all[observe_idx])
# plt.colorbar()
# plt.subplot(2, 1, 2)
# plt.imshow(fea_tag_all[observe_idx, :, :49])
# plt.colorbar()

# plt.figure()
# plt.plot(fea_com_all[observe_idx])

# plt.show()
