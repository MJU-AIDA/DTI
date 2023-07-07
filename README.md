
SumGNN DTI Implementation
========================
[논문명]<<http://example.com/>>
----------------------------


## DRUGBANK
### 
### Drug embedding - DDI
SumGNN provided
> data/drugbank/DB_molecular_feats.pkl

## DAVIS
### Original
> data/davis/Davis_train_origin.csv  
> data/davis/Davis_val_origin.csv  
> data/davis/Davis_test_origin.csv
> data/davis/davis : 위의 세 파일을 단순 copy&paste 

### Only data that can be used with subgraph 
data/davis/Davis_train_test_val.ipynb : network filtering
> data/davis/train.txt  
> data/davis/dev.txt  
> data/davis/test.txt 

### Drug/Target embedding
data/davis/get_dt_pkl.ipynb (코드 리팩토링 중)
> data/davis/DAVIS_drug_feats.pkl  
> data/davis/DAVIS_target_feats.pkl

## KIBA
### Original
> data/kiba/kiba.pkl

### Only data that can be used with subgraph 
> data/kiba/kiba.txt


## Hetionet


## Baseline models
### ML Baseline - SVM, XGBoost, Randomforest
BaselineModel/train_baseline_model.ipynb  

```
python baselinemodel.py 
    -td ./data/davis/Davis_test_origin.csv    # Dataset for Training  
    -vd ./data/davis/Davis_val_origin.csv     # Dataset for Validation  
    -m SVM     #(SVM, XGBoost, RandomForest)  # Choose model to train and see the result. 
```

### DL Baseline - 

## Train
bash run.sh ->
```
python train.py -d davis -e ddi_hop2 --gpu=2 --hop=2 --batch=128 -b=4 --num_epochs 500 --lr 0.00005 
```
