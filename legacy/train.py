import os
import json
import time
import argparse
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

import util, model

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save_model_dir_name', '-s')
parser.add_argument('--save_rate', type=int, default=10)
args = parser.parse_args()

print('Loading data')
fea_tag_all = np.load('fea_tag_all.npy')
fea_com_all = np.load('fea_com_all.npy')
ans_all = np.load('ans_all.npy')
print('Finished, shapes:', fea_tag_all.shape, fea_com_all.shape, ans_all.shape)

print('Removing zeros...')
ans_sum = np.sum(ans_all, axis=1)
keep_idx = np.where(ans_sum>0)[0]
ans_all = ans_all[keep_idx]
fea_com_all = fea_com_all[keep_idx]
fea_tag_all = fea_tag_all[keep_idx]
print('Finished, shapes:', fea_tag_all.shape, fea_com_all.shape, ans_all.shape)

# --- Setup network
nn_param = {
    # For saving how the model was trained
    'batch': 256,
    'optm_params': {
        'lr': 0.001
    },
}

print('Setting network')
network = model.BILSTM_MH_ATTN()
optimizer = optim.Adam(list(network.parameters()), lr=nn_param['optm_params']['lr'])
loss_func = nn.MSELoss()
network.to(device)

# --- Write and copy config
if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))
util.write_cfg(os.path.join(args.save_model_dir_name, 'config.yml'), nn_param)

data_loader = torch.utils.data.DataLoader(
    util.Data2Torch({
        'fea': fea_tag_all,
        'fea_com': fea_com_all,
        'ans': ans_all,
    }),
    shuffle=True,
    batch_size=nn_param['batch'],
)

batch_num = int(fea_tag_all.shape[0] / nn_param['batch'] + 0.5)
accumulation_steps = 8
totalTime = 0
fout = open(os.path.join(args.save_model_dir_name, 'train_report.txt'), 'w')
for epoch in range(args.epoch):
    util.print_and_write_file(fout, 'epoch {}/{}...'.format(epoch + 1, args.epoch))
    tic = time.time()
    # --- Batch training
    network.train()
    training_loss = 0
    n_batch = 0
    optimizer.zero_grad()
    for idx, data in enumerate(data_loader):
        pred = network(
            Variable(data['fea'].to(device)),
            Variable(data['fea_com'].to(device)),
        )
        ans = Variable(data['ans'].to(device))
        loss = loss_func(pred, ans)
        # Normal update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.data
        n_batch += 1
        # # Gradient Accum
        # loss = loss / accumulation_steps
        # loss.backward()
        # if (idx + 1) % accumulation_steps == 0 or (idx + 1) == batch_num:
        #     optimizers[fold].step()
        #     optimizers[fold].zero_grad()
        # training_loss += loss.data
        # n_batch += 1
    # --- Training loss
    training_loss_avg = training_loss / n_batch
    util.print_and_write_file(
        fout, '\tTraining loss (avg over batch): {}, {}, {}'.format(
            training_loss_avg, training_loss, n_batch
        )
    )
    # --- Save if needed
    if (epoch+1) % args.save_rate == 0:
        torch.save(
            network.state_dict(),
            os.path.join(args.save_model_dir_name, 'model_e{}'.format(epoch+1))
        )
    # --- Time
    toc = time.time()
    totalTime += toc - tic
    util.print_and_write_file(fout, '\tTime: {:.3f} sec, estimated remaining: {:.3} hr'.format(
        toc - tic,
        1.0 * totalTime / (epoch + 1) * (args.epoch - (epoch + 1)) / 3600
    ))
    fout.flush()
fout.close()
# Save model
torch.save(
    network.state_dict(),
    os.path.join(args.save_model_dir_name, 'model_final')
)
print('Model saved in {}'.format(args.save_model_dir_name))
