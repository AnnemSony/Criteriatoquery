"""
Created by Sean Edwards
References functions from HuggingFace Token Classification course
2/3/2023
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

import numpy as np
import evaluate
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict

# loads a generic dataset that trains for person, organization,
# location, and miscellaneous entity recognition
loaded_dataset_dict = load_dataset("corpus")
dataset = loaded_dataset_dict["train"]

# create an 80/20 split
train_testvalid = dataset.train_test_split(train_size=0.8)

# split the 20 into 10/10
test_valid = train_testvalid['test'].train_test_split(train_size=0.5)

# gather everything back into new DatasetDict
raw_datasets = DatasetDict(
    {'train': train_testvalid['train'], 'test': test_valid['test'], 'validation': test_valid['train']}
)

def align_labels_with_tokens(labels, word_ids):
    '''
    When tokenizing inputs, the labels may no longer match in length.
    This function is given tokenized input and the labels to match the labels
    to the tokens.

    @labels List of labels for a given example.
    @word_ids The tokenized input

    @return A list of labels that have been padded for the tokenized input
    '''
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

# creating tokenizer from existing model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

def tokenize_and_align_labels(examples):
    '''
    Tokenizes all inputs and uses previously made function to
    align the labels for those tokenized inputs

    @examples The inputs and labels to tokenize and align

    @return the updated dataset to use for training
    '''
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# tokenizing the dataset using batched map for use in training
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# creating a data collator that allows us to pad labels same way as tokenized inputs
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# creating id to label associations for potential inference API support
label_names = raw_datasets["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# creating the model from existing model and giving it the associations
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
    id2label=id2label,
    label2id=label2id,
)

# loading evaluation method
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    '''
    Computes typical metrics for model evaluation

    @eval_preds predictions to evaluate for metrics

    @return dictionary of score per metric
    '''
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

args = TrainingArguments(
    "app/model/bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()