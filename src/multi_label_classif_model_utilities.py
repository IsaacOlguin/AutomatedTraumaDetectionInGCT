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

###############################################################
## GLOBALS
GLB_BERT_MODEL_ID = "Bert"
GLB_BERT_MODEL_NAME = "nlpaueb/legal-bert-small-uncased"

###############################################################
## UTILITIES GENERAL IMPLEMENTATION

"""
["span", "role", "trauma", "court"]
"""
def import_dataset_from_excel(path_dataset, header_index, columns_names_list):
    return pd.read_excel(path_dataset, header=header_index, names=columns_names_list)

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

"""
Function: get_unique_values_from_dataset
"""
def get_unique_values_from_dataset(dataframe, column_name):
    return list(dataframe[column_name].unique())

def get_distribution_classes_from_dataset(dataframe, groupby_list_columns, chosen_column):
    return dataframe.groupby(groupby_list_columns).count()[chosen_column].reset_index()

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
        
"""
Function: get_tokenizer given a model
"""
def get_tokenizer(model_id=GLB_BERT_MODEL_ID, model_name = GLB_BERT_MODEL_NAME, lower_case=True):
    tokenizer = None
    if(model_id == GLB_BERT_MODEL_ID):
        from transformers import BertTokenizer

        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
        print(f'{model_id} tokenizer was loaded successfully ({model_name})', "\n\t", f"do_lower_case={lower_case}")

    return tokenizer