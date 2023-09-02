import sys
import os
import json
import pickle
import warnings

import pandas as pd
import numpy as np

sys.path.append("/home/max38744/moon_project/DTI/nodefeaturing")
from generate_entity_embedding import generating_pro_feature, generating_drug_feature

warnings.filterwarnings("ignore")

def generate_protein_feature(params):
    df = pd.read_csv(f"data/{params.dataset}/origin_proteins.csv").drop("Unnamed: 0", axis = 1)
    df.columns = ["Target_ID", "Target"]
    # Select unique rows based on the 'Drug' column
    unique_targets_df = df.drop_duplicates(subset=['Target'])
    print(f"# of Unique Targets:{len(unique_targets_df)}")

    pro_df = generating_pro_feature(unique_targets_df, params)
    tmp = pro_df[['Target_ID', 'Target',f"{params.protein_embedding_method.upper()}_Features"]]
    # print(tmp.head())

    target_id_array = tmp['Target_ID'].values
    target_array = tmp['Target'].values
    target_feat = tmp[f'{params.protein_embedding_method.upper()}_Features']

    target_feat = {'Target_ID': target_id_array, 'Target': target_array, f"{params.protein_embedding_method.upper()}_Features": target_feat}


    ''' change Protein ID (UniProtKB) to Gene Id (NCBI gene (formerly Entrezgene) ID) '''
    a = pd.read_csv(f"data/{params.dataset}/convert_table", delimiter= "\t")
    new = []
    new_emb = []
    for x_idx, pro in enumerate(target_feat['Target_ID']) :
        for a_idx, name in enumerate(a["UniProtKB Gene Name ID"]) : 
            if pro == name :
                new.append(a['NCBI gene (formerly Entrezgene) ID'].iloc[a_idx])
                new_emb.append(target_feat[f"{params.protein_embedding_method.upper()}_Features"][x_idx])
                break

    new_x = {}
    new_x["Target_ID"] = new
    new_x[f"{params.protein_embedding_method.upper()}_Features"]= new_emb
    new_x = pd.DataFrame(new_x)
    # print(new_x)

    ''' Convert Gene ID to Encoding number And Build Mapping array for training '''
    with open(f"data/{params.dataset}/entity_drug.json", 'r') as f:
        data = json.load(f)
    enco = []
    for i in new :
        if i in data.keys() :
            enco.append(data[i])
        else : 
            enco.append(np.NaN)
    # print(enco)
    new_x["Gene_enco"] = enco
    new_x = new_x.sort_values(by = "Gene_enco").reset_index(drop=True)
    new_x["mapping_arr"] = range(0,len(new_x))
    vec_pfeat = {"Target_ID" : new_x["Target_ID"].values,
                f"{params.protein_embedding_method.upper()}_Features" : new_x[f"{params.protein_embedding_method.upper()}_Features"],
                "Gene_enco" : new_x["Gene_enco"].values,
                "map_arr" : new_x["mapping_arr"].values}

    with open(f"data/{params.dataset}/VEC_target_feats_{params.protein_embedding_method}.pkl", 'wb') as f:
        pickle.dump(vec_pfeat, f)
    print(f"Dictionary saved as VEC_target_feats_{params.protein_embedding_method}.pkl")


def generate_drug_feature(params):
    ''' drugbankid2smiles '''
    file_path = f"data/{params.dataset}/dti2vec_drugbankid2smiles"
    df = pd.read_csv(file_path, header=None, delimiter="\t")  # Assuming the file has no header row
    print(len(df))
    print(df.iloc[[0,1,300,600,1000]])
    df.columns = ["Drug_ID", "Drug"]

    ''' Load json file for mapping encoding & Drug ID '''
    temp = []
    json_file = f"data/{params.dataset}/node2id.json"
    with open(json_file) as f:
        data = json.load(f)

    for i_idx, item in enumerate(df["Drug_ID"].copy()):
        for j_idx, d_name in enumerate(data.keys()):
            if d_name == item:
                temp.append(data[d_name])
                break
            elif j_idx +1 == len(data):
                temp.append(np.NaN)

    ''' get drug features '''
    # Select unique rows based on the 'Drug' column
    unique_drugs_df = df.drop_duplicates(subset=['Drug']).dropna()
    unique_drugs_df = generating_drug_feature(unique_drugs_df, params)
    # print(len(unique_drugs_df))
    # print(unique_drugs_df)
    
    # Need to be modified by params level
    tmp = unique_drugs_df[['Drug_ID', 'Drug','MORGAN_Features']]
    drug_id_array = tmp['Drug_ID'].values
    drug_array = tmp['Drug'].values
    drug_feat = tmp['MORGAN_Features']
    drug_enco = temp

    drug_feat = {'Drug_ID': drug_id_array, 'Drug': drug_array, 'MORGAN_Features': drug_feat, "Drug_enco" : drug_enco}
    with open(f"data/{params.dataset}/VEC_drug_feats_MORGAN.pkl", 'wb') as f:
        pickle.dump(drug_feat, f)
    print(f"Dictionary saved as pickle file")