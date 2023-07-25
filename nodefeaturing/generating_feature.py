from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

### Node feature embedding
def generating_pro_feature(dataset) :
    # Load ProtBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
    def protein_sequence_to_embedding(sequence):
        import re
        # sequence_Example = "A E T C Z A O"
        sequence = ' '.join(char for char in sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        # print(sequence)
        # Tokenize the protein sequence
        encoded_input = tokenizer(sequence, return_tensors='pt')
        #Generate the embedding vector
        outputs = model(**encoded_input)
        # print(outputs)
        # Extract the embedding vector
        embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).detach().numpy()
        return embedding
    
    print(dataset.columns)
    unique_values = dataset[['Target_ID', 'Target']].value_counts().index

    my_id = []
    my_protein = []

    for i in unique_values :
        my_id.append(i[0])
        my_protein.append(i[1])

    # protein = pd.concat([pd.DataFrame( {"Target" : my_protein}),(pd.DataFrame( {"Target_ID" : my_id}))], axis=1)
    # protein['ProtBERT_Features'] = protein['Target'].apply(protein_sequence_to_embedding)
    # Create an empty DataFrame to store the results
    
    protein = pd.DataFrame()

    # Iterate over unique protein sequences and generate embeddings
    for _, row in dataset.iterrows():
        sequence = row['Target']
        embedding = protein_sequence_to_embedding(sequence)

        # Append the results to the DataFrame
        row['ProtBERT_Features'] = embedding
        protein = protein.append(row)

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
    # 'Drug' 열의 값에 대한 Morgan fingerprint 생성
    d['Morgan_Features'] = d['Drug'].apply(smiles_to_fingerprint)
    # print(d['Morgan_Features'])
    # print(d.head())
    return d

def concat_feature(drug_table, target_table) : 
    
    result = pd.merge(drug_table, target_table, on='Target_ID', how='left')

    return result

if __name__ == '__main__':
    # Load ProtBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
    def protein_sequence_to_embedding(sequence):
        import re
        # sequence_Example = "A E T C Z A O"
        sequence = ' '.join(char for char in sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        # print(sequence)
        encoded_input = tokenizer(sequence, return_tensors='pt')
        outputs = model(**encoded_input)
        print(outputs)
        # # Tokenize the protein sequence
        # inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
        # # Generate the embedding vector
        # with torch.no_grad():
        #     outputs = model(**inputs)
        # Extract the embedding vector
        embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).detach().numpy()
        return embedding

    pro_seq = "MLAFILSRATPRPALGPLSYREHRVALLHLTHSMSTTGRGVTFTINCSGFGQHGADPTALNSVFNRKPFRPVTNISVPTQVNISFAMSAILDVNEQLHLLSSFLWLEMVWDNPFISWNPEECEGITKMSMAAKNLWLPDIFIIELMDVDKTPKGLTAYVSNEGRIRYKKPMKVDSICNLDIFYFPFDQQNCTLTFSSFLYTVDSMLLDMEKEVWEITDASRNILQTHGEWELLGLSKATAKLSRGGNLYDQIVFYVAIRRRPSLYVINLLVPSGFLVAIDALSFYLPVKSGNRVPFKITLLLGYNVFLLMMSDLLPTSGTPLIGVYFALCLSLMVGSLLETIFITHLLHVATTQPPPLPRWLHSLLLHCNSPGRCCPTAPQKENKGPGLTPTHLPGVKEPEVSAGQMPGPAEAELTGGSEWTRAQREHEAQKQHSVELWLQFSHAMDAMLFRLYLLFMASSIITVICLWNT"
    pro_seq = ' '.join(char for char in pro_seq)

    embedding = protein_sequence_to_embedding(pro_seq)
    print()
    print(embedding)