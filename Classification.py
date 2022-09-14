import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import collections
# Parts need to be adjusted to load correct model and data


# Import data
# Csv files containing texts and labels (separate files), both for training and test datasets

train_texts = pd.read_csv('') # FILL IN
train_texts = list(train_texts['train_texts'])

train_labels = pd.read_csv('') #FILL IN
train_labels = list(train_labels['train_labels'])

test_texts = pd.read_csv('') # FILL IN
test_texts = list(test_texts['test_texts'])

test_labels = pd.read_csv('') # FILL IN
test_labels = list(test_labels['test_labels'])

target_names = ['Non-arg', 'Impact sceptic', 'Attribution sceptic', 'Trend sceptic', 'No consensus', 'Bad science', 'Conspiracy', 'AGW']

##### DATA PREP #####


# Tokenizer for RobBERT
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base") # ORINAL RobBERT
# Encoding texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=200)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=200)

# Relabelling annotation for smooth training and prediction
tr_labels = []
te_labels = []
for i in range(len(train_labels)):
    if train_labels[i] ==0:
        tr_labels.append(0)
    elif train_labels[i] ==11:
        tr_labels.append(1)
    elif train_labels[i] ==12:
        tr_labels.append(2)
    elif train_labels[i] ==13:
        tr_labels.append(3)
    elif train_labels[i] ==14:
        tr_labels.append(4)
    elif train_labels[i] ==15:
        tr_labels.append(5)
    elif train_labels[i] ==16:
        tr_labels.append(6)
    else:
        tr_labels.append(7)
        
for i in range(len(test_labels)):
    if test_labels[i] ==0:
        te_labels.append(0)
    elif test_labels[i] ==11:
        te_labels.append(1)
    elif test_labels[i] ==12:
        te_labels.append(2)
    elif test_labels[i] ==13:
        te_labels.append(3)
    elif test_labels[i] ==14:
        te_labels.append(4)
    elif test_labels[i] ==15:
        te_labels.append(5)
    elif test_labels[i] ==16:
        te_labels.append(6)
    else:
        te_labels.append(7)
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = Dataset(train_encodings, tr_labels)
test_dataset = Dataset(test_encodings, te_labels)        

# Load model 
# Pick model (saved locally!)

# Not finetuned:
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robBERT-base",num_labels=len(target_names))

# V1
#model = RobertaForSequenceClassification.from_pretrained("robbert/Robbert-CC-level2", num_labels=len(target_names))

# V2
#model = RobertaForSequenceClassification.from_pretrained("robbert/Robbert-CC-level2-v2", num_labels=len(target_names))


# V3
#model = RobertaForSequenceClassification.from_pretrained("robbert/Robbert-CC-level2-v5",num_labels=len(target_names))

# Trainer specifics

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Evaluation metrics
    acc = accuracy_score(labels, preds)   
    f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'f1': f1,  
      'precision': precision,
      'recall': recall  
          }

training_args = TrainingArguments(
    output_dir='',          # make dir
    num_train_epochs=10,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=16,   
    warmup_steps=100, 
    learning_rate=5e-5,
    weight_decay=0.01,               
    logging_dir='',            # make dir
    load_best_model_at_end=True,     
    metric_for_best_model='f1',
    logging_steps=100,               
    evaluation_strategy="steps",     
)
trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=test_dataset,         
    compute_metrics=compute_metrics, 
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training
trainer.train()

# Evaluation after training
trainer.evaluate()

# Better save than sorry
model_path = ""                # make dir
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Validation

# Easy prediction
def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=250, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]


# Import validation data
validation = pd.read_excel('')


v_sentences = []
for row in validation.itertuples():
    v_sentences.append(row.content)
v_labels = validation.label  

v_predictions = []
for i in range(len(v_sentences)):
    pred = get_prediction(v_sentences[i])
    v_predictions.append(pred)
    

# Labelling the test_label integers equal to relabelled annotations earlier
test_labels = v_labels
te_labels = []
for i in range(len(test_labels)):
    if test_labels[i] ==0:
        te_labels.append(0)
    elif test_labels[i] ==11:
        te_labels.append(1)
    elif test_labels[i] ==12:
        te_labels.append(2)
    elif test_labels[i] ==13:
        te_labels.append(3)
    elif test_labels[i] ==14:
        te_labels.append(4)
    elif test_labels[i] ==15:
        te_labels.append(5)
    elif test_labels[i] ==16:
        te_labels.append(6)
    else:
        te_labels.append(7)

v_labels2 = []
for i in range(len(te_labels)):
    v_labels2.append(target_names[te_labels[i]])

cm = confusion_matrix(v_predictions, v_labels2)
# Metrics per class
print('Prec on validation set:', precision_score(v_labels2, v_predictions, average=None))
print('Recall on validation set:', recall_score(v_labels2, v_predictions, average=None))
print('F1 on validation set:',f1_score(v_labels2, v_predictions, average=None))

# Macro metrics for whole classification
print('F1 on validation set:', f1_score(v_labels2, v_predictions,  average='macro'))
print('Precision on validation set:', precision_score(v_labels2, v_predictions, average='macro'))
print('Recall on validation set:',recall_score(v_labels2, v_predictions, average='macro'))

# Quick confusion matrix plot
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  
ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels'); 
ax.set_title('Confusion Matrix'); 



