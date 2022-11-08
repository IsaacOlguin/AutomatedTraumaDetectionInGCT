###############################################################
## Description: Text segmentation and cleaning
## Author: Isaac Misael Olguin Nolasco
## November 2022, TUM
###############################################################

##=====================================================================
#### Imports

from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import luima_sbd.sbd_utils as sbd_utils
import pandas as pd
import spacy
import datetime
import string

##=====================================================================
#### Global variables

INPUT_FILES_PATH_DIRECTORY = ""
OUTPUT_SEGM_CLEAN_PATH_DIRECTORY = ""
LUIMA_DB_PROJECT_DIR = ""
tokenizer = RegexpTokenizer(r'\w+')
info_files = []
dict_nltk = {}
FILE_ENCODING = "utf8"

list_stop_words_english = stopwords.words("english")
stemmer = WordNetLemmatizer()

##=====================================================================
#### Implementation

##============================

def pre_processing_sentence(sentence_tokens):   
    # LowerCase
    new_sentence = sentence_tokens.lower()
    
    # Tokenization with RegEx
    new_sentence = ' '.join(tokenizer.tokenize(new_sentence))
    
    # Remove punctuation
    new_sentence = [character for character in new_sentence if character not in string.punctuation]
    
    # Remove numbers
    new_sentence = ''.join([ character for character in new_sentence if not character.isdigit() ])
    
    # Remove stopwords
    new_sentence = [ token for token in new_sentence.split() if token not in list_stop_words_english ]
    new_sentence = ' '.join(new_sentence)
    
    # Remove lemmas
    new_sentence = [ stemmer.lemmatize(token) for token in new_sentence.split() ]
    new_sentence = ' '.join(new_sentence)
    
    return new_sentence

##============================

def main():
    print("Starting execution of text segmenter and cleaning of sentences...")
    onlyfiles = []
    num_files = 0
    content = ""
    list_of_sentences = []
    
    # List filenames from directory
    try:
        onlyfiles = [f for f in listdir(INPUT_FILES_PATH_DIRECTORY) if isfile(join(INPUT_FILES_PATH_DIRECTORY, f))]
        num_files = len(onlyfiles)
    except Exception as excep:
        print(f"The following exception occurs while reading the content of the directory <<{INPUT_FILES_PATH_DIRECTORY}>>")
        print(excep)
        return
    try:
        # Iterate files over the list ("onlyfiles")
        for index, filename in enumerate(onlyfiles):
            print(f"Processing file ({(index+1)}/{num_files}) [{filename}]... ", end="")

            print("Reading... ", end="")
            try:
                f = open(INPUT_FILES_PATH_DIRECTORY + filename, "r", encoding=FILE_ENCODING)
                content = f.read()
            except Exception as e:
                print(f"\n ERROR An exception occurrs while reading the file <<{filename}>>")
                continue 

            try:
                print("ExtractingSentences[", end="")
                list_of_sentences = sbd_utils.text2sentences(content)
                print(f"{len(list_of_sentences)}]... ", end="")
            except Exception as e:
                print(f"\n ERROR An exception occurrs while getting list of sentences of the file <<{filename}>>")
                continue 

            print("PreProcessing[", end="")
            # Open file
            file_to_write = open(OUTPUT_SEGM_CLEAN_PATH_DIRECTORY + filename, "w", encoding=FILE_ENCODING)
            #Iterate over each sentece
            for j, sent in enumerate(list_of_sentences):

                cleaned_sent = pre_processing_sentence( sent )
                if cleaned_sent.strip() != "":
                    file_to_write.write(f'{cleaned_sent}\n') 
                    if j%200 == 0:
                        print(f"+", end="")
                    elif j%50 == 0:
                        print(f".", end="")
            print("]")

            # Close file
            file_to_write.close()
            
    except Exception as e:
        print("An exception occurrs while processing files")
    else:
        print("Execution of the process was successfully finished :)")
    finally:
        onlyfiles = []
        num_files = 0

##============================

##=====================================================================
#### Definition main function

if __name__ == "__main__":
    main()
    

