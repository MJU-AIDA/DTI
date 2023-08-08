from transformers import AutoTokenizer, AutoModel          # ProtBERT
from transformers import T5Tokenizer, T5EncoderModel       # ProtT5

import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def generating_pro_feature(dataset) :

# Load ProtBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)

    def protein_sequence_to_embedding(sequence : str,
        len_seq_limit =1024
        ):
        '''
        Function to collect last hidden state embedding vector from pre-trained ProtBERT Model

        INPUTS:
        - sequence (str) : protein sequence (ex : AAABBB) from fasta file
        - len_seq_limit (int) : maximum sequence lenght (i.e nb of letters) for truncation

        OUTPUTS:
        - output_hidden : last hidden state embedding vector for input sequence of length 1024
        '''
        sequence_w_spaces = ' '.join(list(sequence))
        encoded_input = tokenizer(
            sequence_w_spaces,
            truncation=True,
            max_length=len_seq_limit,
            padding='max_length',
            return_tensors='pt').to(device)
        output = model(**encoded_input)
        print(output.last_hidden_state.shape)
        print(output['last_hidden_state'][:,0][0])

        output_hidden = output['last_hidden_state'][:,0][0].detach().cpu().numpy()
        assert len(output_hidden)==1024
        return output_hidden

    unique_values = dataset[['Target_ID', 'Target']].value_counts().index

    my_id = []
    my_protein = []

    for i in unique_values :
        my_id.append(i[0])
        my_protein.append(i[1])

    protein = pd.concat([pd.DataFrame( {"Target" : my_protein}),(pd.DataFrame( {"Target_ID" : my_id}))], axis=1)
    protein['ProtBERT_Features'] = protein['Target'].apply(protein_sequence_to_embedding)
    return protein
    
def generating_drug_feature(drug_dataset) :
    d = drug_dataset
    l =[]
    def smiles_to_fingerprint(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        # Generate 1024-dim Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        # Convert fingerprint to a binary vector
        fingerprint_vector = fingerprint.ToBitString()
        fingerprint_vector = list(fingerprint_vector)
        return fingerprint_vector

    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # print(smiles_to_fingerprint(smiles))
    # 'Drug' column의 값에 대한 Morgan fingerprint 생성
    d['Morgan_Features'] = d['Drug'].apply(smiles_to_fingerprint)
    # print(d['Morgan_Features'])
    # print(d.head())
    return d

def concat_feature(drug_table, target_table) :     
    result = pd.merge(drug_table, target_table, on='Target_ID', how='left')
    return result