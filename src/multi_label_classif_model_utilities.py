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
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import random
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

###############################################################
## GLOBALS
GLB_BERT_MODEL_ID = "Bert"
GLB_BERT_BASE_UNCASED_MODEL_NAME = "bert-base-uncased"#"nlpaueb/legal-bert-small-uncased"
GLB_PYTORCH_TENSOR_TYPE = "pt"
GLB_DEVICE_CPU = "cpu"

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
        device = torch.device(GLB_DEVICE_CPU)
    
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
Function: split_dataset_train_val_test 
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
##==========================================================================================================
"""
Function: create_dataloader
"""
def create_dataloader(dataset, batch_size, sampler=RandomSampler):
    return DataLoader(
            dataset,  # The samples.
            sampler = sampler(dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
##==========================================================================================================
"""
Function: create_model
"""
def create_model(model_id, model_name, _num_classes, runInGpu, _output_attentions=False, _output_hidden_states=False):
    model = None

    if model_id == GLB_BERT_MODEL_ID:
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            model_name, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = _num_classes, # The number of output labels   
            output_attentions = _output_attentions, # Whether the model returns attentions weights.
            output_hidden_states = _output_hidden_states, # Whether the model returns all hidden-states.
        )

    if runInGpu:
        # Tell pytorch to run this model on the GPU.
        model.cuda()

    return model
        
##==========================================================================================================
"""
Function: get_optimizer
"""
def get_optimizer(model, learning_rate = 2e-5, epsilon=1e-8, _weight_decay=0):
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,#2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = epsilon, # args.adam_epsilon  - default is 1e-8.
                  weight_decay=_weight_decay#0.01
                )
    return optimizer
##==========================================================================================================
"""
Function: get_scheduler
"""
def get_scheduler(optimizer, value=600, _min_lr=1e-5):
    scheduler = CosineAnnealingLR(
        optimizer, 
        value, 
        eta_min = _min_lr
    )
    return scheduler
##==========================================================================================================
"""
Function: flat_accuracy
Description: Function to calculate the accuracy of our predictions vs labels
"""
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
##==========================================================================================================
"""
Function: format_time
"""
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
##==========================================================================================================
"""
Function: format_time
"""
def train_and_validate(model, device, num_epochs, optimizer, scheduler, train_dataloader, validation_dataloader):
    tr_metrics = []
    va_metrics = []
    tmp_print_flag = True


    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, num_epochs):
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        io_total_train_acc = 0
        io_total_train_prec = 0
        io_total_train_recall = 0
        io_total_train_f1 = 0
        io_total_valid_acc = 0
        io_total_valid_prec = 0
        io_total_valid_recall = 0
        io_total_valid_f1 = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 100 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)
            """
            if tmp_print_flag:
            tmp_print_flag = False
            print(f'result.keys() = {result.keys()}')
            """

            loss = result.loss
            logits = result.logits

            """
            print(f'loss {loss}')
            print(f'logits {logits}')
            """
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_targets.extend(batch[2].numpy())

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            train_acc = accuracy_score(train_targets, train_preds, average=None)
            train_precision = precision_score(train_targets, train_preds, average=None)
            train_recall = recall_score(train_targets, train_preds, average=None)
            train_f1 = f1_score(train_targets, train_preds, average=None)

            io_total_train_acc += train_acc
            io_total_train_prec += train_precision
            io_total_train_recall += train_recall
            io_total_train_f1 += train_f1

        io_avg_train_acc = io_total_train_acc / len(train_dataloader)
        io_avg_train_prec = io_total_train_prec / len(train_dataloader)
        io_avg_train_recall = io_total_train_recall / len(train_dataloader)
        io_avg_train_f1 = io_total_train_f1 / len(train_dataloader)
        print(
            f'Epoch {epoch_i+1} : \n\
            Train_acc : {io_avg_train_acc}\n\
            Train_F1 : {io_avg_train_f1}\n\
            Train_precision : {io_avg_train_prec}\n\
            Train_recall : {io_avg_train_recall}'
        )

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        valid_preds = []
        valid_targets = []

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the 
            # softmax.
            loss = result.loss
            logits = result.logits

            valid_preds.extend(logits.argmax(dim=1).cpu().numpy())
            valid_targets.extend(batch[2].numpy())

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to(GLB_DEVICE_CPU).numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
            valid_acc = accuracy_score(valid_targets, valid_preds, average=None)
            valid_precision = precision_score(valid_targets, valid_preds, average=None)
            valid_recall = recall_score(valid_targets, valid_preds, average=None)
            valid_f1 = f1_score(valid_targets, valid_preds, average=None)

            io_total_valid_acc += valid_acc
            io_total_valid_prec += valid_precision
            io_total_valid_recall += valid_recall
            io_total_valid_f1 += valid_f1

        io_avg_valid_acc = io_total_valid_acc / len(validation_dataloader)
        io_avg_valid_prec = io_total_valid_prec / len(validation_dataloader)
        io_avg_valid_recall = io_total_valid_recall / len(validation_dataloader)
        io_avg_valid_f1 = io_total_valid_f1 / len(validation_dataloader)
        print(
                f'Epoch {epoch_i+1} : \n\
                Valid_acc : {io_avg_valid_acc}\n\
                Valid_F1 : {io_avg_valid_f1}\n\
                Valid_precision : {io_avg_valid_prec}\n\
                Valid_recall : {io_avg_valid_recall}'
            )

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Accur.': io_avg_train_acc,
                'Training F1': io_avg_train_f1,
                'Training Precision': io_avg_train_prec, 
                'Training Recall': io_avg_train_recall,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Valid. F1': io_avg_valid_f1,
                'Valid. Precision': io_avg_valid_prec, 
                'Valid. Recall': io_avg_valid_recall,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return model
