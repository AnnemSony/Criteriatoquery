# Refactored C2Q

Refactoring the Criteria2Query project using Python with Flask and updated models implemented in Python.

# Directory Structure

This repository contains code from other repositories for various modules which could not be reorganized for import and referencing reasons, so the following section
documents what each directory is associated with.
```
corpus/
- Dataset class and processed dataset used for HuggingFace NER model

dataset/
- Unprocessed datasets as well as tool to process them for corpus class

tools/
- Various tools to convert original dataset format into various formats needed for each module.

app/
- General Flask files + relationship extraction training script.

app/additional_models/
- Base models for relationship extraction module

app/data/
- Finetuned model for relationship extraction

app/src/
- Code associated to the training and inference functions of the relationship extraction module

app/model/
- Location of finetuned NER model and negation detection model

app/lib/
- Files that instantiate each module and provide functions to interact with them.
-- A notable exception is the negation detection module, as its code is located in app/run.py, is instantiated in app/views.py, and has its functions in app/lib/neg_cues.py

app/static/
- Static resources for Flask website.

app/templates/
- HTML files for Flask website.
```

# Setup

Create a Python 3.9.13 virtual environment using following commands.
```
pip install virualenv
python -m virtualenv -p=3.9.13 venv
```

Activate the virtual environment then install the required packages using the following command.
```
pip install -r requirements.txt
```

All model downloads provided will require extracting the contents of the internal folder into a specific directory with a specific name for proper functionality.

For a pretrained NER model, download model below and extract it into `app/model/bert-finetuned-ner`.
https://drive.google.com/file/d/1xxSiggtqbKSvGvgvqQtIICoA1Ums1Xvw/view?usp=share_link

To train NER model, run the following commands to run the pipeline script which will process the provided datasets into a usable format for NER model training.
```
cd dataset
python preprocess.py
```

Run `ner_train.py` within the root directory of the project using the following commands to create the model used for NER.
```
cd ..
python ner_train.py
```

Download the negation detection model from the original Criteria2Query 2.0 and put it into `app/model/negdetect/model`.
https://drive.google.com/file/d/1uBbSL0_Zp70Z4vMlAQq43qlRhmg5cIq0/view

Download base relation extraction model below and extract it into `app/additional_models/biobert_v1.1_pubmed`.
https://drive.google.com/file/d/1PQhj4oOTatwg5GUQ2bsTY747Vk17kwnB/view?usp=sharing

For pretrained relationship extraction model, download finetuned model below and extract it into `app/data`. The following model was trained with all relationships from Chia and Covid-19 datasets without blank relations. If using this model, change line 5 of `app/lib/rex.py` to have `num_classes=28`.
https://drive.google.com/file/d/1Oi2zd1Xaftw3YdAn1ls7WbRegPpDTIu9/view?usp=sharing

An alternate pretrained relationship extraction model is provided below which was trained off of only `has_temporal` and `has_value` relations, but with blank `none` relations provided for every other entity pair. If using this model, change line 5 of `app/lib/rex.py` to have `num_classes=6`.
https://drive.google.com/file/d/1u4Rtw8OD2yhfaYbpliWrsvIVwxqK_oEN/view?usp=sharing

To manually train relation extraction, change directory to `tools/brat2semeval` and run the following command to convert brat standoff to Semeval Task 8 2010 styled data.
```
python format_convertor.py
```

Change directory to `app` and run the following command to train the relation extraction model.
```
python relationship_extraction_train.py --train_data ../tools/brat2semeval/train.txt --test_data ../tools/brat2semeval/test.txt --model_no 2 --model_size "bert-base-uncased" --train 1 --batch_size 4 --num_classes 6
```

Depending on your device having only CPU support or wanting to enable GPU support, you may need to add or remove `map_location="cpu"` to any calls of `torch.load()`. Some instances potentially relevant to running of this project can be found in the following places.
- `app/run.py` line 22 - Negation Detection 
- `app/src/model/BERT/modeling_utils.py` line 395 - Relation Extraction
- `app/src/tasks/infer.py` line 287 - Relation Extraction
- `app/src/tasks/train_funcs.py` line 29 and 32 - Relation Extraction
- `app/src/tasks/trainer.py` line 99 - Relation Extraction

Set up an OMOP CDM PostgreSQL database. The SynPUF 1k dataset and scripts for database creation can be found here:
https://drive.google.com/file/d/1G43U3Ip-zqeHRu-feIdLsjzxc_k9Ax54/view?usp=share_link

Steps to create the database are below:
1. First create an empty PostgreSQL database schema.
2. Execute the script `OMOP CDM ddl - PostgreSQL.sql` to create the tables and fields.
3. Load your data into the schema. This is done by using commands such as below for every CSV file provided in the download above.
```SQL
\COPY PERSON FROM 'person.csv' DELIMITER E'\t' CSV ENCODING 'UTF8';
```
4. Execute the script `OMOP CDM constraints - PostgreSQL.sql` to add the constraints (primary and foreign keys).
5. Execute the script `OMOP CDM indexes required - PostgreSQL.sql` to add the minimum set of indexes recommended.


Create a `app/lib/config.py` file that looks as follows:
```python
DB_PASSWORD = "********"
DB_USER = "postgres"
DB_DATABASE = "synpuf1k"
DB_HOST = "localhost" # "34.70.212.14"
```

Run the Flask server with all models by changing directory to `app` and running the following command
```
python run.py
```

# Accreditation

Created with the work done on previous Criteria2Query iterations (https://github.com/OHDSI/Criteria2Query) as well as borrowing the negation detection model used there.

Included in the repository is standoff2conll (https://github.com/spyysalo/standoff2conll) slightly modified for specific use with the Chia dataset.

Also included as part of the repository is a PyTorch implementation of Relation Extraction for BERT by plkmo (https://github.com/plkmo/BERT-Relation-Extraction). Slight modifications were performed to allow it to run with the virtual environment used.