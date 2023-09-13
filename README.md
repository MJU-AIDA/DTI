
SumGNN-DTI - Drug Target Interaction Prediction with Knowledge Graph
========================
![모델아키텍쳐230913](https://github.com/MJU-AIDA/DTI/assets/91061904/53f622c1-96e7-4467-a87c-ae1a4f01ae6b)
----------------------------

# External Knowledge Graph Dataset - Hetionet
[HETIONET](https://het.io/)
> data/hetio/metaedge_encoding.json : Only the encoding value included in the original relations_2hop.txt exists
> 
# Dataset
## DRUGBANK
> data/drugbank/train.txt  
> data/drugbank/dev.txt  
> data/drugbank/test.txt  
> data/drugbank/drugbank.txt

### Drug embedding - DDI
SumGNN provided
> data/drugbank/DB_molecular_feats.pkl

## DAVIS
### Original
> data/davis/Davis_train_origin.csv  
> data/davis/Davis_val_origin.csv  
> data/davis/Davis_test_origin.csv  
> data/davis/davis : 위의 세 파일을 단순 concat  

### Only data that can be used with subgraph 
data/davis/Davis_train_test_val.ipynb : network filtering
> data/davis/train.txt  
> data/davis/dev.txt  
> data/davis/test.txt  
> data/davis/davis.txt : input for hetio extractor  

### Drug/Target embedding
data/davis/get_dt_pkl.ipynb (코드 리팩토링 중)
> data/davis/DAVIS_drug_feats.pkl  
> data/davis/DAVIS_target_feats.pkl

## KIBA
### Original
> data/kiba/kiba.pkl

### Only data that can be used with subgraph 
> data/kiba/train.txt  
> data/kiba/dev.txt  
> data/kiba/test.txt  
> data/kiba/kiba.txt : input for hetio extractor

# Baseline models
### ML Baseline - SVM, XGBoost, Randomforest
BaselineModel/train_baseline_model.ipynb  

```
python baselinemodel.py 
    -td ./data/davis/Davis_test_origin.csv    # Dataset for Training  
    -vd ./data/davis/Davis_val_origin.csv     # Dataset for Validation  
    -m SVM     #(SVM, XGBoost, RandomForest)  # Choose model to train and see the result. 
```

# DL Baseline
## HyperAttentionDTI

# Train
bash run.sh ->
```
python train.py -d davis -e ddi_hop2 --gpu=2 --hop=2 --batch=128 -b=4 --num_epochs 500 --lr 0.00005 
```
