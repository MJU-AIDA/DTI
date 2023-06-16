import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse
import os
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('../FeatureExtract')
from generating_feature import generating_pro_feature, generating_drug_feature, concat_feature
sys.path.remove('../FeatureExtract')

parser = argparse.ArgumentParser()
parser.add_argument("-td", "--train_dataset", type=str, required=True, help="path to input train dataset")
parser.add_argument("-vd", "--validation_dataset", type=str, required=True, help="path to input validation dataset")
parser.add_argument("-m", "--model", type=str, required=True, help="model to run")

args = parser.parse_args()

train_table = pd.read_csv(args.train_dataset)
val_table = pd.read_csv(args.validation_dataset)
model_name = args.model




train_pro_table = generating_pro_feature(train_table)

train_drug_table = generating_drug_feature(train_table)

train_merge_on = concat_feature(train_drug_table, train_pro_table)

val_pro_table = generating_pro_feature(val_table)

val_drug_table = generating_drug_feature(val_table)

val_merge_on = concat_feature(val_drug_table,val_pro_table)


def my_SVM(df_train , df_val , d_col, p_col, r_col):
    # Train
    train_p_feat = pd.DataFrame(df_train[d_col].tolist())
    train_d_feat = pd.DataFrame(df_train[p_col].tolist())
    
    val_p_feat = pd.DataFrame(df_val[d_col].tolist())
    val_d_feat = pd.DataFrame(df_val[p_col].tolist())
    
    
    #print(pd.concat([p_feat, d_feat], axis = 1))
    clf = svm.SVC(kernel='rbf')
    clf.fit(pd.concat([train_p_feat, train_d_feat], axis = 1), df_train[r_col])
    
    # Evaluation
    pred_rels_train = clf.predict(pd.concat([train_p_feat, train_d_feat], axis = 1))
    accuracy_train = accuracy_score(df_train[r_col], pred_rels_train)
    f1score_train = f1_score(df_train[r_col], pred_rels_train)
    
    pred_rels_val = clf.predict(pd.concat([val_p_feat, val_d_feat], axis = 1))
    accuracy_val = accuracy_score(df_val[r_col], pred_rels_val)
    f1score_val = f1_score(df_val[r_col], pred_rels_val)
    
    print("Validation :","Accuracy:", accuracy_val, "F1-score", f1score_val)
    
    print("Train :", "Accuracy:", accuracy_train, "F1-score", f1score_train)


def my_XGBoost(df_train, df_val, d_col, p_col, r_col):
    def prepare_data(df):
        p_feat = pd.DataFrame(df[p_col].tolist())
        d_feat = pd.DataFrame(df[d_col].tolist()).astype(int)
        
        concat_feats = pd.concat([p_feat, d_feat], axis=1)
        gt_rels = df[r_col]
        
        concat_feats.columns = range(concat_feats.shape[1])
        
        return xgb.DMatrix(concat_feats, label=gt_rels)
    
    dtrain = prepare_data(df_train)
    dval = prepare_data(df_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'logloss', 'auc'],
        'max_depth': 3,
        'learning_rate': 0.1
    }
    
    model = xgb.train(params, dtrain)
    
    def evaluate(dmatrix, gt_rels):
        pred = model.predict(dmatrix)
        pred_binary = [1 if p >= 0.5 else 0 for p in pred]
        accuracy = accuracy_score(gt_rels, pred_binary)
        f1score = f1_score(gt_rels, pred_binary)
        return accuracy, f1score
    
    accuracy_train, f1score_train = evaluate(dtrain, df_train[r_col])
    print("Train - Accuracy:", accuracy_train, "F1-score:", f1score_train)
    
    accuracy_val, f1score_val = evaluate(dval, df_val[r_col])
    print("Validation - Accuracy:", accuracy_val, "F1-score:", f1score_val)


def my_RandomForest(df_train, df_val, d_col, p_col, r_col):
    def prepare_data(df):
        p_feat = pd.DataFrame(df[d_col].tolist())
        d_feat = pd.DataFrame(df[p_col].tolist())
        concat_feats = pd.concat([p_feat, d_feat], axis=1)
        gt_rels = df[r_col]
        return concat_feats, gt_rels

    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    def evaluate(X, y):
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        f1score = f1_score(y, pred)
        return accuracy, f1score
    
    accuracy_train, f1score_train = evaluate(X_train, y_train)
    print("Train - Accuracy:", accuracy_train, "F1-score:", f1score_train)
    
    accuracy_val, f1score_val = evaluate(X_val, y_val)
    print("Validation - Accuracy:", accuracy_val, "F1-score:", f1score_val)


if model_name == 'SVM' :
    my_SVM(train_merge_on, val_merge_on,'Morgan_Features', 'ProtBERT_Features', "Y")

elif model_name == 'XGBoost' :
    my_XGBoost(train_merge_on, val_merge_on,'Morgan_Features', 'ProtBERT_Features', "Y")

elif model_name == "RandomForest" :
    my_RandomForest(train_merge_on, val_merge_on,'Morgan_Features', 'ProtBERT_Features', "Y")

else :
    print("we don't have the model, available model : SVM, XGBoost, RandomForest")
