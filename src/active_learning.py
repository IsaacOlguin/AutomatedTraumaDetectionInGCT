"""
###################################################################################################
###################################################################################################
@author:    Isaac Misael Olguín Nolasco
@project:   Automated Trauma Detection in Genocide Court Transcripts (IDP - TUM Informatics)
@file:      Contains the implementation for the active learning process.
###################################################################################################
###################################################################################################
"""

###################################################################################################
###################################################################################################
### Definition of globals
###################################################################################################
###################################################################################################
GLB_DEFINE_PATH_PROJECT = False
PATH_PROJECT = ""
READ_FILE_MODE = "r"
PATH_DATASET = ""
PATH_DIR_LOGS = "logs"
PATH_DIR_MODELS = ""
INDEX_COLUMNS_DATASET = ""
LIST_NAME_COLUMNS_DATASET = ""
GLB_RETURN_ATTENTION_MASK = ""
GLB_ADD_SPECIAL_TOKENS = True
GLB_MAX_LENGTH_SENTENCE = 512
GLB_PADDING_TO_MAX_LENGTH = True
GLB_CROSS_VALIDATION = ""
GLB_SAVE_MODEL = ""
GLB_STORE_STATISTICS_MODEL = ""
GLB_TEST_MODEL = ""
GLB_SIZE_SPLITS_DATASET = 1
COL_OF_INTEREST = ""
CLASSIFICATION_TASK = ""
COL_OF_REFERENCE = ""
GLB_RUN_IN_GPU = True
LOGGER = None
        
###################################################################################################
###################################################################################################
### Imports
###################################################################################################
###################################################################################################
# Required packages
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from os.path import join
import sys
import datetime as dt
import collections
import logging
from sklearn.exceptions import UndefinedMetricWarning
import warnings
# Custom code
import multi_label_classif_model_utilities as mlclassif_utilities


###################################################################################################
###################################################################################################
### Logging
###################################################################################################
###################################################################################################
"""
Function:       configure_logger()
Description:    Configure logger of the project
Return:         None
"""
def configure_logger(levelStdout=logging.DEBUG, levelFile=logging.DEBUG):
    global LOGGER
    
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(levelStdout)
    stdout_handler.setFormatter(formatter)
    """
    
    file_handler = logging.FileHandler(join(PATH_PROJECT, PATH_DIR_LOGS, get_datetime_format() + '_activeLearning.log'))
    
    file_handler.setLevel(levelFile)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    #LOGGER.addHandler(stdout_handler)
    
def infoLog(message):
    if LOGGER != None:
        LOGGER.info(message)
    else: 
        print(f"INFO  {message}")

def debugLog(message):
    if LOGGER != None:
        LOGGER.debug(message)
    else: 
        print(f"DEBUG {message}")
    
def errorLog(message):
    if LOGGER != None:
        LOGGER.error(message)
    else: 
        print(f"ERROR {message}")
    
def warnLog(message):
    if LOGGER != None:
        LOGGER.warn(message)
    else: 
        print(f"WARN  {message}")

###################################################################################################
###################################################################################################
### Control for warnings
###################################################################################################
###################################################################################################
def warn(*args, **kwargs):
    pass
warnings.warn = warn

###################################################################################################
###################################################################################################
### Functions for set-up
###################################################################################################
###################################################################################################

"""
Function:       get_projects_directory_path()
Description:    Defines the path of the project's directory
Return:         None (path is stored in a global variable)
"""
def get_projects_directory_path():
    if GLB_DEFINE_PATH_PROJECT:
        PATH_PROJECT = "/content/drive/MyDrive/Colab Notebooks/AutomatedTraumaDetectionInGCT"
    else:
        PATH_PROJECT = os.getcwd()
        
"""
Function:       read_config_file(config_file_path)
Description:    Reads the configuration file (yaml file)
Return:         None (configuration is stored in global variables)
"""
def read_config_file(config_file_path):
    if config_file_path == None:
        infoLog("Path of the configuration file is not defined. It's considered a default value ['config.yml']")
        config_file_path = "config.yml"
    
    with open(join(PATH_PROJECT, config_file_path), READ_FILE_MODE) as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        debugLog(cfg)
    
        global PATH_DATASET
        PATH_DATASET = join( PATH_PROJECT, cfg["general_set_up"]["input_dir_name"], cfg["general_set_up"]["dataset_dir_name"], cfg["general_set_up"]["dataset_filename"] )
        
        global PATH_DIR_LOGS
        PATH_DIR_LOGS = join( PATH_PROJECT, cfg["general_set_up"]["logs_dir_name"] )
        
        global PATH_DIR_MODELS
        PATH_DIR_MODELS = join( PATH_PROJECT, cfg["general_set_up"]["models_dir_name"] )
        
        global INDEX_COLUMNS_DATASET
        INDEX_COLUMNS_DATASET = cfg["dataset"]["index_columns_dataset"]
        
        global LIST_NAME_COLUMNS_DATASET
        LIST_NAME_COLUMNS_DATASET = cfg["dataset"]["list_columns_names"]
        
        global GLB_RETURN_ATTENTION_MASK
        GLB_RETURN_ATTENTION_MASK = cfg["training_model"]["return_attention_mask"]
        
        global GLB_CROSS_VALIDATION
        GLB_CROSS_VALIDATION = cfg["training_model"]["cross_validation"]
        
        global GLB_SAVE_MODEL
        GLB_SAVE_MODEL = cfg["training_model"]["save_model"]
        
        global GLB_STORE_STATISTICS_MODEL
        GLB_STORE_STATISTICS_MODEL = cfg["training_model"]["store_statistics"]
        
        global GLB_TEST_MODEL
        GLB_TEST_MODEL = cfg["training_model"]["test_model"]
        
        # Globals for the model
        global EPOCHS
        EPOCHS = cfg["training_model"]["epochs"]
        
        global EMBEDDING_SIZE
        EMBEDDING_SIZE = cfg["training_model"]["embedding_size"]
        
        global BATCH_SIZE
        BATCH_SIZE = cfg["training_model"]["batch_size"]
        
        global GLB_ADD_SPECIAL_TOKENS
        GLB_ADD_SPECIAL_TOKENS = cfg["training_model"]["add_special_tokes"]
        
        global GLB_MAX_LENGTH_SENTENCE
        GLB_MAX_LENGTH_SENTENCE = cfg["training_model"]["max_length"]
        
        global GLB_PADDING_TO_MAX_LENGTH
        GLB_PADDING_TO_MAX_LENGTH = cfg["training_model"]["pad_to_max_length"]
        
        global GLB_RUN_IN_GPU
        GLB_RUN_IN_GPU = cfg["training_model"]["run_in_gpu"]
        
        # Active training
        global GLB_SIZE_SPLITS_DATASET
        GLB_SIZE_SPLITS_DATASET = cfg["active_training"]["size_splits_dataset"]
        
        global CLASSIFICATION_TASK
        global COL_OF_INTEREST
        global COL_OF_REFERENCE
        CLASSIFICATION_TASK = cfg["active_training"]["classification_task"]
        if CLASSIFICATION_TASK == "binary":
            COL_OF_INTEREST = cfg["dataset"]["col_of_interest_binary_classif"]
            COL_OF_REFERENCE = cfg["dataset"]["col_of_reference_binary_classif"]
        elif CLASSIFICATION_TASK == "multi":
            COL_OF_INTEREST = cfg["dataset"]["col_of_interest_multi_label_classif"]
            COL_OF_REFERENCE = cfg["dataset"]["col_of_reference_multi_label_classif"]

"""
Function:       read_input_arguments()
Description:    Reads the arguments that user gives when execute this file
Return:         List - of input arguments
"""  
def read_input_arguments():
    return sys.argv

"""
Function:       get_datetime_format()
Description:    Get the date and time with the format for storing files
Return:         String - Date and time
"""
def get_datetime_format():
    return dt.datetime.now().strftime("%Y%m%d%H%M%S")
    
###################################################################################################
###################################################################################################
### Functions for training
###################################################################################################
###################################################################################################
"""
Function:       train(dataframe)
Description:    Execute training
Return:         Tuple - (model, metrics)
""" 
def train(df_dataset, num_classes):
    # Get classes of the dataset
    classes_dataset = mlclassif_utilities.get_unique_values_from_dataset(df_dataset, "role")
    debugLog(f"Num of different roles in the dataset is {len(classes_dataset)} which are:")
    for index, elem in enumerate(classes_dataset):
        debugLog(f"\t {index+1} - {elem}")
    
    # Define device to be used
    device = mlclassif_utilities.get_gpu_device_if_exists()
    debugLog(f"\n\n==> Selected device is '{device}' <==")
    
    #If no parameters are sent, default values are considered. 
    #    IDModel:      Bert
    #    Model namel:  bert-base-uncased
    #    Do uncase:    True
    
    # Get tokenizer
    tokenizer = mlclassif_utilities.get_tokenizer() 

    # Get lists of spans and classes
    list_all_spans = list(df_dataset[COL_OF_REFERENCE])
    list_all_classes = list(df_dataset[COL_OF_INTEREST])
    
    # Get & print the sentence and length of the largest one
    mlclassif_utilities.get_max_length_of_a_sentence_among_all_sentences(tokenizer, list_all_spans, False)
    
    # If _return_attention_mask, a tuple of two lists is given (tensor_of_inputs, tensor_of_attention_masks)
    all_spans_tokenized = mlclassif_utilities.get_all_spans_tokenized(
        mlclassif_utilities.GLB_BERT_MODEL_ID, 
        tokenizer,
        list_all_spans,
        _add_special_tokens = GLB_ADD_SPECIAL_TOKENS, 
        _max_length = GLB_MAX_LENGTH_SENTENCE,
        _pad_to_max_length = GLB_PADDING_TO_MAX_LENGTH,
        _return_attention_mask = GLB_RETURN_ATTENTION_MASK, 
        type_tensors = mlclassif_utilities.GLB_PYTORCH_TENSOR_TYPE
    )
    
    input_ids = None
    attention_masks = None
    
    if GLB_RETURN_ATTENTION_MASK:
        input_ids = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized[0])
        attention_masks = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized[1])
    else:
        input_ids = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized)
    
    numeric_classes = mlclassif_utilities.convert_list_span_classes_into_numeric_values(classes_dataset, list_all_classes)
    numeric_classes = mlclassif_utilities.convert_list_labels_into_pytorch_tensor(numeric_classes)
    
    ### Split dataset
    if not GLB_CROSS_VALIDATION:
        train_labels_corpus, train_input_ids, train_attention_masks, val_labels_corpus, val_input_ids, val_attention_masks, test_labels_corpus, test_input_ids, test_attention_masks = mlclassif_utilities.split_dataset_train_val_test(numeric_classes, input_ids, attention_masks)
    else:
        ### k-Fold
        train_val_corpus_cross_validation, test_corpus_cross_validation = mlclassif_utilities.split_dataset_train_val_test_k_fold(numeric_classes, input_ids, attention_masks, 0.1)
        
    ### Create model
    model = mlclassif_utilities.create_model(
        mlclassif_utilities.GLB_BERT_MODEL_ID,
        mlclassif_utilities.GLB_BERT_BASE_UNCASED_MODEL_NAME,
        num_classes,
        GLB_RUN_IN_GPU #RunInGPU
    )
    
    # Define optimizer
    optimizer = mlclassif_utilities.get_optimizer(model)
    
    # Define scheduler
    scheduler = mlclassif_utilities.get_scheduler(optimizer)
    
    if not GLB_CROSS_VALIDATION:
        train_dataset = mlclassif_utilities.create_tensor_dataset(train_input_ids, train_attention_masks, train_labels_corpus)
        val_dataset = mlclassif_utilities.create_tensor_dataset(val_input_ids, val_attention_masks, val_labels_corpus)
        test_dataset = mlclassif_utilities.create_tensor_dataset(test_input_ids, test_attention_masks, test_labels_corpus)
        
        train_dataloader = mlclassif_utilities.create_dataloader(train_dataset, BATCH_SIZE)
        val_dataloader = mlclassif_utilities.create_dataloader(val_dataset, BATCH_SIZE)
        test_dataloader = mlclassif_utilities.create_dataloader(test_dataset, BATCH_SIZE)
        
        model, statistics_model = mlclassif_utilities.train_and_validate(model, device, EPOCHS, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())
    
    else:
        for index_cross_val in range(len(train_val_corpus_cross_validation)):
            train_dataset = mlclassif_utilities.create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][1], train_val_corpus_cross_validation[index_cross_val][2], train_val_corpus_cross_validation[index_cross_val][0])
            val_dataset = mlclassif_utilities.create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][4], train_val_corpus_cross_validation[index_cross_val][5], train_val_corpus_cross_validation[index_cross_val][3])
    
            train_dataloader = mlclassif_utilities.create_dataloader(train_dataset, BATCH_SIZE)
            val_dataloader = mlclassif_utilities.create_dataloader(val_dataset, BATCH_SIZE)
    
            debugLog('='*50)
            debugLog(f"Cross-Validation Split {(index_cross_val+1)}/{len(train_val_corpus_cross_validation)}")
            debugLog('='*50)
            model, statistics_model = mlclassif_utilities.train_and_validate(model, device, EPOCHS, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())
    
    if GLB_STORE_STATISTICS_MODEL:
        mlclassif_utilities.save_json_file_statistics_model(statistics_model, PATH_DIR_LOGS)
    
    if GLB_TEST_MODEL:
        mlclassif_utilities.test_model(model, device, test_dataloader, numeric_classes.tolist())
    
    if GLB_SAVE_MODEL:
        mlclassif_utilities.save_model(model, get_datetime_format() + "_model_bert_" + num_classes + "_classes", PATH_DIR_MODELS)
        
    return model, statistics_model
    
###################################################################################################
###################################################################################################
### Functions for active training
###################################################################################################
###################################################################################################

def give_me_segments_of_df_per_class(df, number_of_splits, column_of_interest, column_of_reference):
    dict_of_segments = {}
    invalidSplit = False
    number_of_classes = df[column_of_interest].nunique()
    list_of_classes = df[column_of_interest].unique()
    
    counts = df[column_of_interest].value_counts()
    normalized = round(df[column_of_interest].value_counts(normalize=True), 4)
    percentages = normalized*100
    
    df_stats_dataset = pd.DataFrame({'counts': counts, 'normalized': normalized, 'percentages': percentages}).reset_index()
    
    # Validation
    for i, row in df_stats_dataset.iterrows():
        if row["counts"] < number_of_splits:
            errorLog(f"ERROR - Dataset[{row['index']}] cannot be split into the given number of splits")
            invalidSplit = True
    
    if invalidSplit:
        return None
    else:
        # Get sizes of segments and put them into a list
        list_of_size_segments = (df_stats_dataset["counts"]-(df_stats_dataset["counts"]%number_of_splits)) / number_of_splits
        
        """
        print("*"*100)
        print(df_stats_dataset)
        print("*"*100)
        """
        
        # Initialize dict_of_segments
        for i_range in range(0, number_of_splits):
            dict_of_segments[i_range] = pd.DataFrame()
        
        # Add segments to a list of segments
        for index_class, (size, type_id) in enumerate(zip(list_of_size_segments, df_stats_dataset["index"])):
            size = int(size)
            #print(index_class, "#"*100, size)
            for i_range in range(0, number_of_splits):
                #print(i_range, "*"*50, index_class, type_id, "Segment", i_range, "[", i_range*size, ":", i_range*size+size, "]")
                if index_class == 0:
                    dict_of_segments[i_range] = df[df[COL_OF_INTEREST] == type_id][i_range*size:i_range*size+size]
                else:
                    if (i_range+1) == number_of_splits:
                        dict_of_segments[i_range] = pd.concat([dict_of_segments[i_range], df[df[COL_OF_INTEREST] == type_id][i_range*size:]])
                    else:
                        dict_of_segments[i_range] = pd.concat([dict_of_segments[i_range], df[df[COL_OF_INTEREST] == type_id][i_range*size:i_range*size+size]])
    
    return dict_of_segments

###################################################################################################
###################################################################################################
### Main function
###################################################################################################
###################################################################################################

def main():
    configure_logger()
    mlclassif_utilities.setLogger(LOGGER)
    
    list_statistics = list()
    
    # Get project's path
    debugLog("Reading directory path")
    get_projects_directory_path()
    # Read configuration file
    read_config_file("config.yml")
    
    # Get input arguments
    #input_arguments = read_input_arguments()
    #print(f"The number of input arguments is [{len(input_arguments)}] whose content is {input_arguments}")
    
    df_dataset = mlclassif_utilities.import_dataset_from_excel(PATH_DATASET, INDEX_COLUMNS_DATASET, LIST_NAME_COLUMNS_DATASET)
    classes_dataset = mlclassif_utilities.get_unique_values_from_dataset(df_dataset, COL_OF_INTEREST)
    
    # Split the dataset for the Active Training purposes
    dict_of_segments = give_me_segments_of_df_per_class(df_dataset, GLB_SIZE_SPLITS_DATASET, COL_OF_INTEREST, COL_OF_REFERENCE)
    odlst = collections.OrderedDict(sorted(dict_of_segments.items()))
    for index, df_at_index in odlst.items():
        if index == 0:
            df = dict_of_segments[index]
        else:
            df = pd.concat([df, dict_of_segments[index]])
        
        # Read config file again - in case it is desired to store a model or change something at execution time
        read_config_file("config.yml")
        
        model, statistics = train(df, len(classes_dataset))
        debugLog("="*100)
        debugLog("*"*80)
        infoLog(statistics)
        debugLog("*"*80)
        debugLog("="*100)
        
if __name__ == "__main__":
    main()