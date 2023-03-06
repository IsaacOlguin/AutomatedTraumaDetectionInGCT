# AutomatedTraumaDetectionInGCT
Repository of the "Automated Trauma Detection in Genocide Court Transcripts" project

As explained by Schirmer et al. in "Uncovering Trauma in Genocide Tribunals: An NLP Approach Using the Genocide Transcript Corpus", this project aims to use NLP techniques to uncover and analyze exposure to trauma to which witnesses in international criminal tribunals are subjected when recounting their experiences in court.

This repository contains the source code in its dev-version that has been developed as a collaboration between the TUM School of Computation, Information and Technology (CIT) and the TUM School of Governance at the Technical University of Munich (TUM) for the Interdisciplinary Project (IDP) required for the program of my master in Informatics (Computer Science).

The image below shows the pipeline followed by the project.
![Pipeline of the IDP](doc_resources/images/IDP-Pipeline.png)

## Directory content

The whole structure of the project is as follows:
* **01_WebScraping_TribunalTranscriptCases.ipynb** Notebook that contains the implementation for scraping the trascripts from cases. It contains three big sections depending on the desired court. In addition to that, it is possible to retrieve only one transcript or multiple transcripts that may be given in an excel file. Files that are the input of the web-scraping processes may be located anywhere and user has to specified them, although it is suggested to locate them into the "input/" directory. Files that are result of these processes are stored into "output/clean_transcripts/[court]" where court can be "eccc", "ictr", or "icty". (This notebook covers the first three steps: "Input", "WebScraping", and "Format cleaning").
* **02_InfoExtractionFromJson.ipynb** Notebook that has the implementation for extracting the information of cases, their sentences and corresponding labels; and store them into an excel file that will be used for the classification process. (This notebook covers part of the fourth step "Segmentation and labeling").
* **03_BinaryClassification.ipynb** Notebook that executes the binary classification located in "src/binary_classification.py" (This notebook covers part of the fifth step "Classification").
* **04_Multiclass_Classification_Model.ipynb** Notebook that executes the multiclass classification (outdated version)
* **05_ActiveLearning.ipynb** Notebook that executes the active learning implementation located in "src/active_learning.py" (This notebook covers part of the fifth step "Classification").
* **06_AnalysisOfComparableModels.ipynb** Notebook that contains the implementation for extracting information from JSON statistic files and plot them in order to compare performances of the conducted experiments (This notebook covers part of the sixth step "Analysis of Comparable Models").
* **07_XAI_Shap.ipynb** Notebook that contains the implementation for using Shap in order to obtain information that allows us to make conclusions about the behaviour of the model as well as inferences of what it/they are learning and therefore, the reasons of their predictions. (This notebook covers part of the seventh step "Analysis of Comparable Models").
* **config.yml** File that contains all configurable parameters required by the implementation in order to be executed. User does not need to make any change in the current source implementation (unless that is desired) but changes depend on the only configuration of this file.
* **src/**
  - **active_learning.py**
  - **binary_classification.py**
  - **classification_model_utilities.py**
  - **cleaning_transcripts.py**
  - **general_utilities.py**
  - **multilabel_classification.py**
  - **utilities_project.py**
* **input/**
  - **json** Directory where the json file has to be located in order to get the information that will be used for the classification task.
  - **dataset** Directory where the dataset for the classification task has to be located. (Once the information is extracted from the JSON file, the process also stores the Dataset here under the name Dataset.xlsx)
* **logs/** Directory where log files are stored.
* **output/**
  - **clean_transcripts**
    - **eccc** Directory where pre-processed files, which belong to the ECCC court belong, are stored.
    - **ictr** Directory where pre-processed files, which belong to the ICTR court belong, are stored.
    - **icty** Directory where pre-processed files, which belong to the ICTY court belong, are stored.
  - **xai**
* **models/** Directory where models are stored.

## Pipeline description

### Input

### WebScraping

### Format cleaning

### Segmentation and labeling

### Classification

### Comparable models' analysis

### Explainability AI (XAI)

## Additionals

### Active Learning
![Active Learning 01](doc_resources/images/ActLrng01.png)
![Active Learning 02](doc_resources/images/ActLrng02.png)
![Active Learning 03](doc_resources/images/ActLrng03.png)
