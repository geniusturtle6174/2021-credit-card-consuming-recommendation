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

以下將介紹本競賽所使用的執行環境、特徵截取、模型設計與訓練。

### 執行環境

硬體方面，初始時使用 ASUS P2440 UF 筆電，含 i7-8550U CPU 及 MX130 顯示卡，主記憶體擴充至 20 GB；後續使用較多特徵及較長期間的資料時，改為使用 AWS p2.xlarge 機器，含 K80 顯示卡以及約 64 GB 主記憶體。AWS 的經費來源是[上一個比賽](https://tbrain.trendmicro.com.tw/Competitions/Details/15)進入複賽拿到的點數，在打完複賽後還有剩下來的部分。

程式語言為 Python 3，未特別指定版本；函式庫則如本說明前半部所示，其中的 matplotlib 為繪圖觀察用，而 yaml 為儲存模型組態用。

### 特徵截取(附帶資料觀察)與預測目標

我先將欄位分為兩類，依照「訓練資料欄位說明」的順序，從 shop_tag（消費類別）起至 card_other_txn_amt_pct （其他卡片消費金額佔比）止，因為是從每月每類的消費行為而來，且消費行為必然是變動的，因此列為「時間變化類」；而 masts （婚姻狀態）起至最後為止，因所觀察到的每人的婚姻狀態或教育程度等，在比賽資料所截取的兩年間幾乎都不會變化，故列為「時間不變類」，以節省運算及儲存資源。事實上，在「時間不變類」的欄位當中，平均每人用過的不同狀態，平均約為 1.005 至 1.167 種，最多的則為 3 至 5 種。

#### 時間變化類

對於每人每月的消費紀錄，以如下步驟取特徵
1. 排序出消費金額前 n 大者，最佳成績中使用的 n 為 13。根據觀察，約 99% 的人，其每月消費類別數在 13 以下。
2. 取該月時間特徵，為待預測月減去該月，共 1 維。
3. 該月類別特徵共 49 維，若該月該類別消費金額在該月前 n 名中且金額大於 0 者，其特徵值由名次大到小依次為 n, n-1, n-2, …, 1；前 n 名以外或金額小於等於 0 的類別，其特徵值為 0。
4. 對於前 n 名的每個類別，無論其消費金額皆取以下特徵，共 22 維：txn\_cnt, txn\_amt, domestic\_offline\_cnt, domestic\_online\_cnt, overseas\_offline\_cnt, overseas\_online\_cnt, domestic\_offline\_amt\_pct, domestic\_online\_amt\_pct, overseas\_offline\_amt\_pct, overseas\_online\_amt\_pct, card\_\*\_txn\_cnt (* = 1, 2, 4, 6, 10, other), card\_\*\_txn\_amt\_pct (\* = 1, 2, 4, 6, 10, other)。
   * 1, 2, 4, 6, 10, other 為所有消費紀錄中，使用次數最多的前六個卡片編號。
5. 以上共 1 + 49 + 13 \* 22 = 336 維

跨月份的取值方式如下圖所示，其中每個圓角方塊代表每人的一個月份的所有消費紀錄，而 N<sub>1</sub> 為 20 個月，N<sub>2</sub> 為 4 組，在範圍內會盡可能的取長或多。另，若該月未有消費紀錄，則忽略該月。

![時間變化類取值方式](images/fea_ext.png "時間變化類取值方式")

#### 時間不變類

對於每人僅使用取值範圍內最後消費當月（N<sub>1</sub> 範圍內的最後一筆）的金額最大的類別所記載的資料來組成特徵。

使用時，以 masts, gender_code, age, primary_card, slam 各自編成 one-hot encoding 或數值型態後組合，共得 20 維，細節說明如下
* masts: 含缺值共 4 種狀態，4 維。
* gender\_code: 含缺值共 3 種狀態，3 維。
* age: 含缺值共 10 種狀態，10 維。
* primary\_card: 沒有缺值，共 2 種狀態，2 維。
* slam: 數值型態，取 log 後做為特徵，1 維。

此部分亦嘗試過其他特徵，但可能是因為維度較大不易訓練（如 cuorg，含缺值共 35 維），或客戶有可能填寫不實（如 poscd），故未取得較好之結果。

#### 預測目標

共 16 維，代表需要預測的 16 個類別，其中下月金額第一名者為 1，第二名者 0.8，第三名者 0.6，第四名以下有購買者 0.2，未購買者 0。

#### 小結

以上取法經去除輸出全部為 0 （即預測目標月份沒有購買行為）之資料後，共約 102 萬組。

### 模型設計與訓練

本次比賽使用的模型架構如下圖，主體為 BiLSTM + attention，前後加上適量的 linear layers，其中標色部分為 attention 的做用範圍，最後面的 dense layers 之細部架構則為 (dense 128 + ReLU + dropout 0.1) * 2 + dense 16 + Sigmoid。

![模型架構](images/model_archi.png "模型架構")

訓練方式為 5 folds cross validation，預測時會將五個模型的結果取平均，再依據平均後的排名輸出前三名的類別。細節參數如下，未提及之參數係依照 pytorch 預設值，未進行修改：
* Num of epochs: 100 epochs，若 validation loss 連續 10 個 epochs 未創新低，則提前終止該 fold 的訓練。
* Batch size: 512。
* Loss: MSE。
* Optimizer: ADAM with learning rate 0.01。
* Learning rate scheduler: 每個 epoch 下降為上一次的 0.95 倍，直至其低於 0.0001 為止。
