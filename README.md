# 2021-credit-card-consuming-recommendation

My implementation and sharing of this contest: https://tbrain.trendmicro.com.tw/Competitions/Details/18. I got rank 9 in the Private Leaderboard.

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

以下將介紹本競賽所使用的特徵截取、模型設計與訓練，以及執行環境，最後亦將簡單介紹幾個我認為在一路上做過的實驗中，可能比較重要的那些，以稍微表示我選擇參數時的決策原因，若有不同發現亦歡迎討論指教。

### 執行環境

硬體方面，初始時使用 ASUS P2440 UF 筆電，含 i7-8550U CPU 及 MX130 顯示卡，主記憶體擴充至 20 GB；後續使用較多特徵及較長期間的資料時，改為使用 AWS p2.xlarge 機器，含 K80 顯示卡以及約 64 GB 主記憶體。AWS 的經費來源是[上一個比賽](https://tbrain.trendmicro.com.tw/Competitions/Details/15)進入複賽拿到的點數，在打完複賽後還有剩下來的部分。

程式語言為 Python 3，未特別指定版本；函式庫則如本說明前半部所示，其中的 matplotlib 為繪圖觀察用，而 yaml 為儲存模型組態用。

### 特徵截取(附帶資料觀察)

我先將欄位分為兩類，依照「訓練資料欄位說明」的順序，從 shop_tag（消費類別）起至 card_other_txn_amt_pct （其他卡片消費金額佔比）止，因為是從每月每類的消費行為而來，且消費行為必然是變動的，因此列為「時間變化類」；而 masts （婚姻狀態）起至最後為止，因所觀察到的每人的婚姻狀態或教育程度等，在比賽資料所截取的兩年間幾乎都不會變化，故列為「時間不變類」。事實上，在「時間不變類」的欄位當中，平均每人用過的不同狀態，平均約為 1.17 種，最多的則為 3 至 5 種。

以下取法經去除輸出全部為 0 （及預測目標月份沒有購買行為）之資料後，共約 102 萬組。

#### 時間變化類

對於每人每月的消費紀錄，以如下步驟取特徵
1. 排序出消費金額前 n 大者，最佳成績中使用的 n 為 13。根據觀察，約 99% 的人，其每月消費類別數在 13 以下。
2. 取該月時間特徵，為待預測月減去該月，共 1 維。
3. 該月類別特徵共 49 維，若該月該類別消費金額在該月前 n 名中且金額大於 0 者，其特徵值由名次大到小依次為 n, n-1, n-2, …, 1；前 n 名以外或金額小於等於 0 的類別，其特徵值為 0。
4. 對於前 n 名的每個類別，無論其消費金額皆取以下特徵，共 22 維：txn_cnt, txn_amt, domestic_offline_cnt, domestic_online_cnt, overseas_offline_cnt, overseas_online_cnt, domestic_offline_amt_pct, domestic_online_amt_pct, overseas_offline_amt_pct, overseas_online_amt_pct, card_*_txn_cnt (* = 1, 2, 4, 6, 10, other), card_*_txn_amt_pct (* = 1, 2, 4, 6, 10, other)。
   * 1, 2, 4, 6, 10, other 為所有消費紀錄中，使用次數最多的前六個卡片編號。
5. 以上共 1 + 49 + 13 * 22 = 336 維

跨月份的取值方式如下圖所示，其中 N_1 為 20 個月，N_2 為 4 組，在範圍內會盡可能的取長或多。另，若該月未有消費紀錄，則忽略該月。

![時間變化類取值方式][images/fea_ext.png "時間變化類取值方式"]

#### 時間不變類

