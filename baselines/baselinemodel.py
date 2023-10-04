import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score,precision_recall_curve, auc
import argparse
import os
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.append("/home/wjdtjr980/my_project/mydti/nodefeaturing")
from generate_entity_embedding import generating_pro_feature, generating_drug_feature, concat_feature

def eval_res(y_true, y_prob, model, threshold=0.5):
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Convert probability scores to binary predictions based on the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
    plt.savefig(f"cm_{model}.png")  
    
    print(classification_report(y_true, y_pred))


def my_SVM(df_train , df_val , d_col, p_col, r_col):
    # Train
    train_p_feat = pd.DataFrame(df_train[d_col].tolist())
    train_d_feat = pd.DataFrame(df_train[p_col].tolist())
    
    val_p_feat = pd.DataFrame(df_val[d_col].tolist())
    val_d_feat = pd.DataFrame(df_val[p_col].tolist())
    
    
    #print(pd.concat([p_feat, d_feat], axis = 1))
    clf = svm.SVC(kernel='rbf', probability=True)
    print("Training SVM..")
    clf.fit(pd.concat([train_p_feat, train_d_feat], axis = 1), df_train[r_col])
    
    # Predict probability scores
    prob_rels_train = clf.predict_proba(pd.concat([train_p_feat, train_d_feat], axis=1))[:, 1]  # Probability of class 1
    prob_rels_val = clf.predict_proba(pd.concat([val_p_feat, val_d_feat], axis=1))[:, 1]  # Probability of class 1
    
    # Evaluation
    accuracy_train = accuracy_score(df_train[r_col], (prob_rels_train > 0.5).astype(int))
    f1score_train = f1_score(df_train[r_col], (prob_rels_train > 0.5).astype(int))
    auroc_train = roc_auc_score(df_train[r_col], prob_rels_train)
    
    precision, recall, _ = precision_recall_curve(df_train[r_col], prob_rels_train)
    auprc_train = auc(recall, precision)

    accuracy_val = accuracy_score(df_val[r_col], (prob_rels_val > 0.5).astype(int))
    f1score_val = f1_score(df_val[r_col], (prob_rels_val > 0.5).astype(int))
    auroc_val = roc_auc_score(df_val[r_col], prob_rels_val)
    
    precision, recall, _ = precision_recall_curve(df_val[r_col], prob_rels_val)
    auprc_val = auc(recall, precision)

    eval_res(df_val[r_col], prob_rels_val, "SVM") 
    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} auROC: {auroc_train:<15.5f} auPRC: {auprc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} auROC: {auroc_val:<15.5f} auPRC: {auprc_val:<15.5f}")

    



def my_XGBoost(df_train, df_val, d_col, p_col, r_col):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
    
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
        'eval_metric': ['logloss'],
        'max_depth': 3,
        'learning_rate': 0.1
    }
    
    print("Training XGB")
    model = xgb.train(params, dtrain)
    
    def evaluate(dmatrix, gt_rels):
        pred = model.predict(dmatrix)
        accuracy = accuracy_score(gt_rels, (pred >= 0.5).astype(int))
        f1score = f1_score(gt_rels, (pred >= 0.5).astype(int))
        auroc = roc_auc_score(gt_rels, pred)
        
        precision, recall, _ = precision_recall_curve(gt_rels, pred)
        auprc = auc(recall, precision)
        
        eval_res(gt_rels, pred, "XGB")  # Use probability scores for eval_res
        
        return accuracy, f1score, auroc, auprc
    
    accuracy_train, f1score_train, auroc_train, auprc_train = evaluate(dtrain, df_train[r_col])
    accuracy_val, f1score_val, auroc_val, auprc_val = evaluate(dval, df_val[r_col])
    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} auROC: {auroc_train:<15.5f} auPRC: {auprc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} auROC: {auroc_val:<15.5f} auPRC: {auprc_val:<15.5f}")


def my_RandomForest(df_train, df_val, d_col, p_col, r_col):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc

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
        prob_pred = model.predict_proba(X)[:, 1]  # Probability of class 1
        accuracy = accuracy_score(y, (prob_pred >= 0.5).astype(int))
        f1score = f1_score(y, (prob_pred >= 0.5).astype(int))
        auroc = roc_auc_score(y, prob_pred)

        precision, recall, _ = precision_recall_curve(y, prob_pred)
        auprc = auc(recall, precision)

        eval_res(y, (prob_pred >= 0.5).astype(int), "RF")  # Use probability scores for eval_res

        return accuracy, f1score, auroc, auprc

    accuracy_train, f1score_train, auroc_train, auprc_train = evaluate(X_train, y_train)
    accuracy_val, f1score_val, auroc_val, auprc_val = evaluate(X_val, y_val)

    print(f"Train   ->   Accuracy: {accuracy_train:<15.5f} F1-score: {f1score_train:<15.5f} auROC: {auroc_train:<15.5f} auPRC: {auprc_train:<15.5f}")
    print(f"Val     ->   Accuracy: {accuracy_val:<15.5f} F1-score: {f1score_val:<15.5f} auROC: {auroc_val:<15.5f} auPRC: {auprc_val:<15.5f}")

print("Input the arguments..")
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

args.gpu = 2
args.protein_embedding_method = "prot_t5_xl_bfd"
args.drug_embedding_method = "morgan"
args.protein_embedding_replace = True
if model_name not in ['SVM', 'XGBoost','RandomForest']:
    print("we don't have the model, available model : SVM, XGBoost, RandomForest")
    exit(0)

print("generating features..")
train_pro_table = generating_pro_feature(train_table, args)

train_drug_table = generating_drug_feature(train_table, args)

train_merge_on = concat_feature(train_drug_table, train_pro_table)

val_pro_table = generating_pro_feature(val_table, args)

val_drug_table = generating_drug_feature(val_table, args)

val_merge_on = concat_feature(val_drug_table,val_pro_table)

print("training..")
if model_name == 'SVM' :
    my_SVM(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")

elif model_name == 'XGBoost' :
    my_XGBoost(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")

elif model_name == "RandomForest" :
    my_RandomForest(train_merge_on, val_merge_on,f"{args.protein_embedding_method.upper()}_Features", f"{args.protein_embedding_method.upper()}_Features", "Y")