import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import argparse
import os
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.append("/home/wjdtjr980/my_project/mydti/nodefeaturing")
from generate_entity_embedding import generating_pro_feature, generating_drug_feature, concat_feature

def eval_res(y_true, y_pred, model):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn.metrics import precision_score, recall_score, f1_score
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
    plt.savefig(f"cm_{model}.png")  
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))


def my_SVM(df_train , df_val , d_col, p_col, r_col):
    # Train
    train_p_feat = pd.DataFrame(df_train[d_col].tolist())
    train_d_feat = pd.DataFrame(df_train[p_col].tolist())
    
    val_p_feat = pd.DataFrame(df_val[d_col].tolist())
    val_d_feat = pd.DataFrame(df_val[p_col].tolist())
    
    
    #print(pd.concat([p_feat, d_feat], axis = 1))
    clf = svm.SVC(kernel='rbf')
    print("Training SVM")
    clf.fit(pd.concat([train_p_feat, train_d_feat], axis = 1), df_train[r_col])
    
    # Evaluation
    pred_rels_train = clf.predict(pd.concat([train_p_feat, train_d_feat], axis = 1))
    accuracy_train = accuracy_score(df_train[r_col], pred_rels_train)
    f1score_train = f1_score(df_train[r_col], pred_rels_train)
    auc_train = roc_auc_score(df_train[r_col], pred_rels_train)

    pred_rels_val = clf.predict(pd.concat([val_p_feat, val_d_feat], axis = 1))
    accuracy_val = accuracy_score(df_val[r_col], pred_rels_val)
    f1score_val = f1_score(df_val[r_col], pred_rels_val)
    auc_val = roc_auc_score(df_val[r_col], pred_rels_val)
    eval_res(df_val[r_col], pred_rels_val, "SVM") 
    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} AUC: {auc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} AUC: {auc_val:<15.5f}")

    



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
    
    print("Training XGB")
    model = xgb.train(params, dtrain)
    
    def evaluate(dmatrix, gt_rels):
        pred = model.predict(dmatrix)
        pred_binary = [1 if p >= 0.5 else 0 for p in pred]
        accuracy = accuracy_score(gt_rels, pred_binary)
        f1score = f1_score(gt_rels, pred_binary)
        auc = roc_auc_score(gt_rels, pred_binary)
        eval_res(gt_rels, pred_binary, "XGB") 
        return accuracy, f1score, auc
    
    accuracy_train, f1score_train, auc_train = evaluate(dtrain, df_train[r_col])
    accuracy_val, f1score_val, auc_val = evaluate(dval, df_val[r_col])
    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} AUC: {auc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} AUC: {auc_val:<15.5f}")


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
    print("Training RF")
    model.fit(X_train, y_train)
    
    def evaluate(X, y):
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        f1score = f1_score(y, pred)
        auc = roc_auc_score(y, pred) 
        eval_res(y, pred, "RF") 
        return accuracy, f1score, auc
    
    accuracy_train, f1score_train, auc_train = evaluate(X_train, y_train)
    accuracy_val, f1score_val, auc_val = evaluate(X_val, y_val)

    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} AUC: {auc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} AUC: {auc_val:<15.5f}")

parser = argparse.ArgumentParser()
parser.add_argument("-td", "--train_dataset", type=str, required=True, help="path to input train dataset")
parser.add_argument("-vd", "--validation_dataset", type=str, required=True, help="path to input validation dataset")
parser.add_argument("-m", "--model", type=str, required=True, help="model to run")

args = parser.parse_args()
if True:
    train_table = pd.read_csv(args.train_dataset)
    val_table = pd.read_csv(args.validation_dataset)
    model_name = args.model
else:
    train_table = pd.read_csv(args.train_dataset, sep=" ", header=None)
    val_table = pd.read_csv(args.validation_dataset, sep=" ", header=None)
    model_name = args.model

args.gpu = 3
args.protein_embedding_method = "prot_bert_bfd"
args.drug_embedding_method = "morgan"
args.protein_embedding_replace = True

train_pro_table = generating_pro_feature(train_table, args)

train_drug_table = generating_drug_feature(train_table, args)

train_merge_on = concat_feature(train_drug_table, train_pro_table)

val_pro_table = generating_pro_feature(val_table, args)

val_drug_table = generating_drug_feature(val_table, args)

val_merge_on = concat_feature(val_drug_table,val_pro_table)


if model_name == 'SVM' :
    my_SVM(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")

elif model_name == 'XGBoost' :
    my_XGBoost(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")

elif model_name == "RandomForest" :
    my_RandomForest(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")

else :
    print("we don't have the model, available model : SVM, XGBoost, RandomForest")
