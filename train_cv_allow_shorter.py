import argparse
import os
import time
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

import model_allow_shorter
import util

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save_model_dir_name', '-s')
parser.add_argument('--n_fold', type=int, default=5)
parser.add_argument('--batch', type=int, default=512)
args = parser.parse_args()

fea_tag_all = []
fea_com_last_all = []
ans_all = []
for key in range(1, 25):
    if not os.path.exists('ans_all_{}.npy'.format(key)):
        continue
    print('Loading data for key', key)
    fea_tag_one_len = np.load('fea_tag_all_{}.npy'.format(key))
    fea_com_last_one_len = np.load('fea_com_last_all_{}.npy'.format(key))
    ans_one_len = np.load('ans_all_{}.npy'.format(key))
    print('\tFinished, shapes:', fea_tag_one_len.shape, fea_com_last_one_len.shape, ans_one_len.shape)
    print('\tRemoving zeros...')
    ans_sum = np.sum(ans_one_len, axis=1)
    keep_idx = np.where(ans_sum > 0)[0]
    ans_one_len = ans_one_len[keep_idx]
    fea_com_last_one_len = fea_com_last_one_len[keep_idx]
    fea_tag_one_len = fea_tag_one_len[keep_idx]
    print('\tFinished, shapes:', fea_tag_one_len.shape, fea_com_last_one_len.shape, ans_one_len.shape)
    print('\tnp array to list...')
    for f_tag, f_com, ans in zip(fea_tag_one_len, fea_com_last_one_len, ans_one_len):
        fea_tag_all.append(f_tag)
        fea_com_last_all.append(f_com)
        ans_all.append(ans)

del fea_tag_one_len, fea_com_last_one_len, ans_one_len
fea_tag_all = np.array(fea_tag_all, dtype=object)
fea_com_last_all = np.array(fea_com_last_all)
ans_all = np.array(ans_all)
print('Overall shapes:', fea_tag_all.shape, fea_com_last_all.shape, ans_all.shape)

# --- Setup network
nn_param = {
    # For saving how the model was trained
    'batch': args.batch,
    'optm_params': {
        'lr': 0.001
    },
}

print('Setting network')
networks = {}
optimizers = {}
schedulers = {}
loss_func = nn.MSELoss()
for f in range(args.n_fold):
    networks[f] = model_allow_shorter.BILSTM_MH_ATTN()
    networks[f].to(device)
    optimizers[f] = optim.Adam(list(networks[f].parameters()), lr=nn_param['optm_params']['lr'])
    schedulers[f] = torch.optim.lr_scheduler.StepLR(optimizers[f], step_size=1, gamma=0.95)

# --- Write and copy config
if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))
util.write_cfg(os.path.join(args.save_model_dir_name, 'config.yml'), nn_param)

data_num = fea_tag_all.shape[0]
for fold in range(args.n_fold):
    best_va_loss = 9999

    valid_idx = np.where(np.arange(data_num) % args.n_fold == fold)[0]
    train_idx = np.where(np.arange(data_num) % args.n_fold != fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    data_loader_train = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_tag_all[train_idx],
            'fea_com_last': fea_com_last_all[train_idx],
            'ans': ans_all[train_idx],
        }),
        shuffle=True,
        batch_size=nn_param['batch'],
        collate_fn=util.collate_fn,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_tag_all[valid_idx],
            'fea_com_last': fea_com_last_all[valid_idx],
            'ans': ans_all[valid_idx],
        }),
        batch_size=nn_param['batch'],
        collate_fn=util.collate_fn,
    )

    va_not_imporved_continue_count = 0
    totalTime = 0
    fout = open(os.path.join(args.save_model_dir_name, 'train_report_{}.txt'.format(fold)), 'w')
    for epoch in range(args.epoch):
        util.print_and_write_file(fout, 'epoch {}/{}...'.format(epoch + 1, args.epoch))
        tic = time.time()
        # --- Batch training
        networks[fold].train()
        training_loss = 0
        n_batch = 0
        optimizers[fold].zero_grad()
        for idx, (data, seq_len) in enumerate(data_loader_train):
            pred = networks[fold](
                Variable(data['fea'].to(device)),
                Variable(data['fea_com_last'].to(device)),
                seq_len,
            )
            ans = Variable(data['ans'].to(device))
            loss = loss_func(pred, ans)
            optimizers[fold].zero_grad()
            loss.backward()
            optimizers[fold].step()
            training_loss += loss.data
            n_batch += 1
        # --- Training loss
        training_loss_avg = training_loss / n_batch
        util.print_and_write_file(
            fout, '\tTraining loss (avg over batch): {}, {}, {}'.format(
                training_loss_avg, training_loss, n_batch
            )
        )
        # --- Batch validation
        networks[fold].eval()
        va_loss = 0
        n_batch = 0
        for idx, (data, seq_len) in enumerate(data_loader_valid):
            ans = Variable(data['ans'].to(device))
            with torch.no_grad():
                pred = networks[fold](
                    Variable(data['fea'].to(device)),
                    Variable(data['fea_com_last'].to(device)),
                    seq_len,
                )
                loss = loss_func(pred, ans)
            va_loss += loss.data
            n_batch += 1
        # --- Validation loss
        va_loss_avg = va_loss / n_batch
        util.print_and_write_file(
            fout, '\tValidation loss (avg over batch): {}, {}, {}'.format(
                va_loss_avg, va_loss, n_batch
            )
        )
        # --- Save if needed
        if va_loss_avg < best_va_loss:
            best_va_loss = va_loss_avg
            va_not_imporved_continue_count = 0
            util.print_and_write_file(fout, '\tWill save bestVa model')
            torch.save(
                networks[fold].state_dict(),
                os.path.join(args.save_model_dir_name, 'model_{}_bestVa'.format(fold))
            )
        else:
            va_not_imporved_continue_count += 1
            util.print_and_write_file(fout, '\tva_not_imporved_continue_count: {}'.format(va_not_imporved_continue_count))
            if va_not_imporved_continue_count >= 10:
                break
        util.print_and_write_file(fout, '\tLearning rate used for this epoch: {}'.format(schedulers[fold].get_last_lr()[0]))
        if schedulers[fold].get_last_lr()[0] >= 1e-4:
            schedulers[fold].step()
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
        networks[fold].state_dict(),
        os.path.join(args.save_model_dir_name, 'model_{}_final'.format(fold))
    )
    print('Model saved in {}'.format(args.save_model_dir_name))
