import torch
import pandas as pd
import numpy as np
import re
import gc
from transformers import BertTokenizer, BertModel         # protbert, prot_bert_bdf
from transformers import T5Tokenizer, T5EncoderModel       # ProstT5, prot_t5_xl_bfd, prot_t5_xl_uniref50
from transformers import logging
from rdkit import Chem
from rdkit.Chem import AllChem

logging.set_verbosity_error()

# def generating_pro_feature(dataset):
def generating_pro_feature(dataset, params):
    gc.collect()
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    device = torch.device(f"cuda:{params.gpu}" if torch.cuda.is_available() else "cpu")
    if params.protein_embedding_method in ["prot_bert", "prot_bert_average"]:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
    elif params.protein_embedding_method == "prot_bert_bfd":
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
    elif params.protein_embedding_method == "prot_t5_xl_bfd":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd").to(device)
    elif params.protein_embedding_method == "prot_t5_xl_uniref50":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    elif params.protein_embedding_method == "prostt5":
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    def protein_sequence_to_embedding(sequence, len_seq_limit=1024):
        # sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        if params.protein_embedding_replace:
            sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        else:
            sequence = " ".join(list(sequence))
        if True: 
            encoded_input = tokenizer(sequence,truncation=True,max_length=len_seq_limit,
                                      padding='max_length',return_tensors='pt').to(device)
            with torch.no_grad():
                output = model(**encoded_input)
                if params.protein_embedding_method == "prot_bert":
                    output_hidden = output['last_hidden_state'][:, 0][0].detach().cpu().numpy() # 지금까지 방식 : row 0
                    # print(output['last_hidden_state'][:, 0][0].detach().cpu().numpy().shape) # (1024,)
                    # print(type(output_hidden)) # <class 'numpy.ndarray'>
                    # output_hidden = np.diagonal(output['last_hidden_state'][0].detach().cpu().numpy()) # diagonal
                elif params.protein_embedding_method == "prot_bert_average":
                    output_hidden = torch.mean(output['last_hidden_state'][0].float(), dim=1).detach().cpu().numpy() # average
                else:
                    output_hidden = output['last_hidden_state'][:, 0][0].detach().cpu().numpy() # 지금까지 방식 : row 0
            assert len(output_hidden) == 1024
        else: # 
            id = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
            input_id = torch.tensor(id['input_ids']).to(device)
            attention_mask = torch.tensor(id['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_id,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
            output_hidden = embedding[0][:seq_len-1][0]
            assert len(output_hidden) == 1024
        return output_hidden
    unique_values = dataset[['Target_ID', 'Target']].value_counts().index
    my_id = []
    my_protein = []
    for i in unique_values:
        my_id.append(i[0])
        my_protein.append(i[1])
    protein = pd.concat([pd.DataFrame({"Target": my_protein}), (pd.DataFrame({"Target_ID": my_id}))], axis=1)
    # protein['PROT_T5_XL_UNIREF50_Features'] = protein['Target'].apply(protein_sequence_to_embedding)
    protein[f'{params.protein_embedding_method.upper()}_Features'] = protein['Target'].apply(protein_sequence_to_embedding)

    return protein

# def generating_drug_feature(drug_dataset) :
def generating_drug_feature(drug_dataset, params) :
    d = drug_dataset
    l =[]
    def smiles_to_fingerprint(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        # Generate 1024-dim Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        # Convert fingerprint to a binary vector
        fingerprint_vector = fingerprint.ToBitString()
        fingerprint_vector = np.array(fingerprint_vector)
        return fingerprint_vector

    ### generate embedding value of Col 'Drug'
    # d['MORGAN_Features'] = d['Drug'].apply(smiles_to_fingerprint)
    d[f'{params.drug_embedding_method.upper()}_Features'] = d['Drug'].apply(smiles_to_fingerprint)
    return d

def concat_feature(drug_table, target_table) :     
    result = pd.merge(drug_table, target_table, on='Target_ID', how='left')
    return result


if __name__ == "__main__":
    print('This is generate_entity_embedding.py')