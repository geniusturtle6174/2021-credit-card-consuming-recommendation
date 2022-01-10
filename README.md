# 2021-credit-card-consuming-recommendation

https://tbrain.trendmicro.com.tw/Competitions/Details/18

## Run My Implementation

### Required libs

matplotlib, numpy, pytorch, and yaml. Versions of them are not restricted as long as they're new enough.

### Preprocess
```bash
python3 data_to_pkl.py
```
* The officially provided csv file should be in `data` dir.
* Output pkl file is also in `data` dir.

### Feature Extraction
```bash
python3 pkl_to_fea_allow_shorter.py
```
* See "作法分享" for detailed description of optional parameters.

### Training
```bash
python3 train_cv_allow_shorter.py -s save_model_dir
```
* `-s`: where you want to save the trained model.

### Inference

#### Generate model outputs
```bash
python3 test_cv_raw_allow_shorter.py model_dir max_len
```
* `model_dir`: directory of the trained model.
* `max_len`: max number of month considered for each customer.

#### Merge model outputs
```bash
python3 test_cv_merge_allow_shorter.py n_fold_train
```
* `n_fold_train`: number of folds used for training.

## 作法分享


