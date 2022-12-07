###############################################################
## Description: Utilities implementation for the multi-label 
##              classification model
## Author: Isaac Misael Olguín Nolasco
## December 2022, TUM
###############################################################

###############################################################
## IMPORTS

import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split

###############################################################
## GLOBALS
GLB_BERT_MODEL_ID = "Bert"
GLB_BERT_BASE_UNCASED_MODEL_NAME = "bert-base-uncased"#"nlpaueb/legal-bert-small-uncased"
GLB_PYTORCH_TENSOR_TYPE = "pt"

###############################################################
## UTILITIES GENERAL IMPLEMENTATION

##==========================================================================================================
"""
["span", "role", "trauma", "court"]
"""
def import_dataset_from_excel(path_dataset, header_index, columns_names_list):
    return pd.read_excel(path_dataset, header=header_index, names=columns_names_list)
##==========================================================================================================
"""
Function: get_gpu_device_if_exists
"""
def get_gpu_device_if_exists():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.\n\nThese are the available devices:' % torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print('\t', index+1, "-", torch.cuda.get_device_name(index))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device
##==========================================================================================================
"""
Function: get_unique_values_from_dataset
"""
def get_unique_values_from_dataset(dataframe, column_name):
    return list(dataframe[column_name].unique())
##==========================================================================================================
"""
Function: get_distribution_classes_from_dataset
"""
def get_distribution_classes_from_dataset(dataframe, groupby_list_columns, chosen_column):
    return dataframe.groupby(groupby_list_columns).count()[chosen_column].reset_index()
##==========================================================================================================
"""
Function: get_max_length_of_a_sentence_among_all_sentences
"""
def get_max_length_of_a_sentence_among_all_sentences(tokenizer, list_all_sentences, add_special_tokens=True):
    # ==> Get the max length of a sentence
    max_len = 0
    list_length_sentences = list()

    # For every sentence...
    for sentence in list_all_sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sentence, add_special_tokens=add_special_tokens)

        # Update the maximum sentence length.
        list_length_sentences.append(len(input_ids))
        #max_len = max(max_len, len(input_ids))
        
    index_max = list_length_sentences.index(max(list_length_sentences))
    print('Max sentence length: ', list_length_sentences[index_max], "found at index", index_max, ". Sentence is:\n\n\n", list_all_sentences[index_max], "\n\n\n")
    return list_length_sentences[index_max]
##==========================================================================================================
"""
Function: get_tokenizer given a model
"""
def get_tokenizer(model_id=GLB_BERT_MODEL_ID, model_name = GLB_BERT_BASE_UNCASED_MODEL_NAME, lower_case=True):
    tokenizer = None
    if(model_id == GLB_BERT_MODEL_ID):
        from transformers import BertTokenizer

        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
        print(f'{model_id} tokenizer was loaded successfully ({model_name})', "\n\t", f"do_lower_case={lower_case}")

    return tokenizer
##==========================================================================================================
"""
Function: get_all_spans_tokenized
Note: 
 - If tokenizer=get_tokenizer(), it's created another instance of the tokenizer
 - If tokenizer=get_tokenizer, it's not created another instance unless what is sent is not an instance of some model's tokenizer
"""
def get_all_spans_tokenized(model=GLB_BERT_MODEL_ID, tokenizer=get_tokenizer, all_spans=[], _add_special_tokens=True, _max_length=512, _pad_to_max_length = True, _return_attention_mask=True, type_tensors=GLB_PYTORCH_TENSOR_TYPE):
    input_ids = []
    attention_masks = []

    if model == GLB_BERT_MODEL_ID:
        # For every sentence...
        for span in all_spans:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                span,                      # Sentence to encode.
                                add_special_tokens = _add_special_tokens, # Add '[CLS]' and '[SEP]'
                                max_length = _max_length,          # Pad & truncate all sentences.
                                pad_to_max_length = _pad_to_max_length,  #is deprecated
                                return_attention_mask = _return_attention_mask,   # Construct attn. masks.
                                return_tensors = type_tensors,     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            if _return_attention_mask:
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])


    if _return_attention_mask:
        return input_ids, attention_masks
    else:
        return input_ids
##==========================================================================================================
"""
Function: convert_list_into_pytorch_tensor 
"""
def convert_list_into_pytorch_tensor(input_list):
    if type(input_list) != list:
        print(f"Warning! - The input parameter does not correspond to the expected type value. Found {type(input_list)}")
        return None
    
    return torch.cat(input_list, dim=0)
##==========================================================================================================
"""
Function: convert_list_span_classes_into_numeric_values
"""
def convert_list_span_classes_into_numeric_values(list_classes, list_spans):
    dict_classes = {}
    for index, elem in enumerate(list_classes):
        dict_classes[elem] = index

    return [dict_classes[it] for it in list_spans]
##==========================================================================================================
"""
Function: convert_list_labels_into_pytorch_tensor 
"""
def convert_list_labels_into_pytorch_tensor(input_list):
    if type(input_list) != list:
        print(f"Warning! - The input parameter does not correspond to the expected type value. Found {type(input_list)}")
        return None
    
    return torch.tensor(input_list)
##==========================================================================================================
"""
Function: create_dataset 
"""
def create_tensor_dataset(input_ids, attention_masks, labels):
    return TensorDataset(input_ids, attention_masks, labels)
##==========================================================================================================
"""
Function: create_dataset 
"""
def split_dataset_train_val_test(labels, input_ids, attention_masks, test_size_percentage=0.05, val_size_percentage=0.1, debug=True):
    train_valid_indices, test_indices = train_test_split(
        np.arange(len(labels)), 
        test_size=test_size_percentage, 
        shuffle=True, 
        stratify=labels
    )

    # TRAINING AND VALIDATION CORPUS (labels, input_ids, attention_masks)
    train_valid_labels = labels[train_valid_indices]
    train_valid_input_ids = input_ids[train_valid_indices]
    train_valid_attention_masks = attention_masks[train_valid_indices]

    # TEST: (labels, input_ids, attention_masks)
    test_labels_corpus = labels[test_indices] 
    test_input_ids = input_ids[test_indices]
    test_attention_masks = attention_masks[test_indices]

    train_indices, valid_indices = train_test_split(
        np.arange(len(train_valid_labels)), 
        test_size=val_size_percentage, 
        shuffle=True, 
        stratify=train_valid_labels
    )

    # TRAIN: (labels, input_ids, attention_masks)
    train_labels_corpus = train_valid_labels[train_indices] 
    train_input_ids = train_valid_input_ids[train_indices]
    train_attention_masks = train_valid_attention_masks[train_indices]

    # VALIDATION: (labels, input_ids, attention_masks)
    val_labels_corpus = train_valid_labels[valid_indices] 
    val_input_ids = train_valid_input_ids[valid_indices]
    val_attention_masks = train_valid_attention_masks[valid_indices]

    if debug == True:
        print("CORPUS TRAINING AND VALIDATION: ", 
            "\n\t", f"Length labels {len(train_valid_labels)}",
            "\n\t", f"Length input_ids {len(train_valid_input_ids)}",
            "\n\t", f"Length attention_masks {len(train_valid_attention_masks)}",
            "\n"
            )

        print("\tCORPUS TRAINING: ", 
            "\n\t\t", f"Length labels {len(train_labels_corpus)}",
            "\n\t\t", f"Length input_ids {len(train_input_ids)}",
            "\n\t\t", f"Length attention_masks {len(train_attention_masks)}",
            )

        print("\tCORPUS VALIDATION: ", 
            "\n\t\t", f"Length labels {len(val_labels_corpus)}",
            "\n\t\t", f"Length input_ids {len(val_input_ids)}",
            "\n\t\t", f"Length attention_masks {len(val_attention_masks)}",
            )

        print("")

        print("CORPUS TEST: ", 
            "\n\t", f"Length labels {len(test_labels_corpus)}",
            "\n\t", f"Length input_ids {len(test_input_ids)}",
            "\n\t", f"Length attention_masks {len(test_input_ids)}", 
            "\n"
        )

    return train_labels_corpus, train_input_ids, train_attention_masks, val_labels_corpus, val_input_ids, val_attention_masks, test_labels_corpus, test_input_ids, test_attention_masks
    