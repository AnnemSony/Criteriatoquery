# Refactored C2Q (The data and everything are private)

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
The data and everything are private!

```

