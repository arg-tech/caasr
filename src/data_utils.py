# data_processing/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# data_processing/data_util.py
import torch
from torch.nn.functional import pad

def load_and_preprocess_data(data_path, sample_size=None, test_size=0.2, random_state=42):
    df_original = pd.read_csv(data_path)

    if sample_size:
        df = df_original.head(sample_size)
    else:
        df = df_original.copy()

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_data, test_data

def encode_labels(arg_relations):
    return LabelEncoder().fit_transform(arg_relations)



def get_distance_based_positional_embedding(distance, p_dim):
    # Calculate angles for sine and cosine
    angles = torch.arange(0, p_dim, 2.0) * -(1.0 / torch.pow(10000, (torch.arange(0.0, p_dim, 2.0) / p_dim)))

    # Calculate positional embeddings using sine and cosine functions
    positional_embeddings = torch.zeros(1, p_dim)
    positional_embeddings[:, 0::2] = torch.sin(distance * angles)
    positional_embeddings[:, 1::2] = torch.cos(distance * angles)

    return positional_embeddings

def tokenize_and_concat(propo1, propso2, argument, relation, tokenizer,relation_freq,
                        prop0_involved_in_other_relations):
    # Implementation of tokenize_and_concat
    arguments = f"[BG]   {argument}  [EG]" 
    propos = f"[BP1]  {propo1}  [EP1]  [BP2]  {propso2} [EP2]"
    propos_p = f"{propo1}  [SEP]  {propso2} [SEP]"

    max_size = 800
    tokenized_props = [tokenizer(propos, return_tensors="pt", padding=True, truncation=True)]
    tokenized_texts = [tokenizer(arguments, return_tensors="pt", padding=True, truncation=True)]

    token_ids = [encoded["input_ids"] for encoded in tokenized_texts]
    attention_masks = [encoded["attention_mask"] for encoded in tokenized_texts]
    props_token_ids = [encoded["input_ids"] for encoded in tokenized_props]
    

    tokenized_input = tokenizer(propos, truncation=True, max_length=max_size, padding='max_length', return_tensors="pt")


    max_len = max_size
    # Calculate distance between propo1 and propso2
    sentences = argument.split("'[SEP]'")  # Assuming sentences are separated by periods
    sentences = [sentence.strip() for sentence in sentences]
    propo1 = propo1.strip()
    propso2 = propso2.strip()
    prop1_index = sentences.index(propo1)
    prop2_index = sentences.index(propso2)
    distance = abs(prop1_index - prop2_index)
    
    #print(len(sentences), prop1_index, prop2_index, distance, relation)
    
    # Increment the frequency count for the current relation and distance
    overlabed_counted = False
    if relation in relation_freq:
        if distance in relation_freq[relation]:
            relation_freq[relation][distance] += 1
        else:
            relation_freq[relation][distance] = 1
    else:
        relation_freq[relation] = {distance: 1}
        
        
        # Check if the propositions involved in relation 0 have already been considered in other relations
    if relation != 0:
        if (propo1, propso2) in prop0_involved_in_other_relations:
            prop0_involved_in_other_relations[(propo1, propso2)].append(relation)
        else:
            prop0_involved_in_other_relations[(propo1, propso2)] = [relation]

    # Check if the propositions involved in relation 0 have been involved in other relations

    else:##
        for key in prop0_involved_in_other_relations:
            if propo1 in key or propso2 in key:
                overlabed_counted = True
        if overlabed_counted:                
            overlapped_distances = relation_freq[0].get("overlapped_0_with_others", {})
            if "overlapped_0_with_others" in relation_freq:
                if distance in relation_freq["overlapped_0_with_others"]:
                    relation_freq["overlapped_0_with_others"][distance] += 1
                else:##
                    relation_freq["overlapped_0_with_others"][distance] = 1 
            else:#
                relation_freq["overlapped_0_with_others"] = {distance: 1}

    positions = torch.tensor(distance).unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]

    p_dim = 10
    positional_embeddings = get_distance_based_positional_embedding(positions, p_dim)


    return (
        tokenized_input,
        pad(positional_embeddings, (0, 128 - positional_embeddings.shape[1]), value=0),  # Assuming pad value is 0
        relation_freq,
        prop0_involved_in_other_relations
    )


def prepare_inputs(data, tokenizer):
    # Implementation of prepare_inputs
    tokenized_results = []
    position_embeddings = []
    micro_labels =  []
    arguments = data['argument']
    prop_1_texts = data['prop_1']
    prop_2_texts = data['prop_2']
    relations = data['label']
    
    # Define relation_freq dictionary to store frequency count for each relation for each distance
    relation_freq = {}
    prop0_involved_in_other_relations = {}

    for prop_1, prop_2, argument,relation in zip(prop_1_texts, prop_2_texts, arguments, relations):
        tokenized_input, position_embedding,relation_freq,prop0_involved_in_other_relations = tokenize_and_concat(prop_1, prop_2, argument, 
                                                                                relation,tokenizer,
                                                                                relation_freq,
                                                                               prop0_involved_in_other_relations)
        tokenized_results.append(tokenized_input)
        position_embeddings.append(position_embedding)
        
    # Sort the relation_freq dictionary based on distances from largest to smallest for each relation
    sorted_relation_freq = {}
    for relation, distances in relation_freq.items():
        sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[0], reverse=True)}
        sorted_relation_freq[relation] = sorted_distances

    print(relation_freq)
        
    return tokenized_results, position_embeddings


