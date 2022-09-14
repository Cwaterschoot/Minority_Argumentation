import pandas as pd
import numpy as np
import collections
from modAL.models import ActiveLearner
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
import math
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import classifier_uncertainty
from modAL.disagreement import max_disagreement_sampling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib as mpl
import matplotlib.pyplot as plt


df2 = pd.read_csv('') # Csv with texts
labels = pd.read_csv('') # Csv with labels
labels = list(labels['train_labels'])
df2['label'] = labels
df2 = df2.rename(columns={'train_texts': 'content', 'label': 'label'})

######## EMBEDDING EXTRACTION #######
sentences = []
for row in df2.itertuples():
    sentences.append(row.content)
labels = df2.label   


word_embedding_model = models.Transformer("robbert/Robbert-CC-level2") # Fill in model to extract embeddings from

# Apply mean pooling or take CLS vector to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

sentence_embeddings = model.encode(sentences)

emd = pd.DataFrame(sentence_embeddings)

# Assign datasets to use during committee training
x_train = emd
y_train = labels

# During active learning, we delete samples from the dataset. That is why we make duplicates of the entire data.
temp = emd
temp_y = y_train

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
temp = temp.to_numpy()
temp_y = temp_y.to_numpy()

####### COMMITTEE CREATION #########

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import classifier_uncertainty
from modAL.disagreement import max_disagreement_sampling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random


# initializing Committee members
learner_list = list()


random.seed(1234)
# initial training data
n_initial = 10



# Starting datasets for 5 learners


train_idx1 = np.random.choice(range(x_train.shape[0]), size=n_initial, replace=False)
x_pool1 = x_train[train_idx1]
y_pool1 = y_train[train_idx1]
# creating a reduced copy of the data with the known instances removed
x_train = np.delete(x_train, train_idx1, axis=0)
y_train = np.delete(y_train, train_idx1)

train_idx2 = np.random.choice(range(x_train.shape[0]), size=n_initial, replace=False)
x_pool2 = x_train[train_idx2]
y_pool2 = y_train[train_idx2]
# creating a reduced copy of the data with the known instances removed
x_train = np.delete(x_train, train_idx2, axis=0)
y_train = np.delete(y_train, train_idx2)

train_idx3 = np.random.choice(range(x_train.shape[0]), size=n_initial, replace=False)
x_pool3 = x_train[train_idx3]
y_pool3 = y_train[train_idx3]
# creating a reduced copy of the data with the known instances removed
x_train = np.delete(x_train, train_idx3, axis=0)
y_train = np.delete(y_train, train_idx3)

train_idx4 = np.random.choice(range(x_train.shape[0]), size=n_initial, replace=False)
x_pool4 = x_train[train_idx4]
y_pool4 = y_train[train_idx4]
# creating a reduced copy of the data with the known instances removed
x_train = np.delete(x_train, train_idx4, axis=0)
y_train = np.delete(y_train, train_idx4)

train_idx5 = np.random.choice(range(x_train.shape[0]), size=n_initial, replace=False)
x_pool5 = x_train[train_idx5]
y_pool5 = y_train[train_idx5]
# creating a reduced copy of the data with the known instances removed
x_train = np.delete(x_train, train_idx5, axis=0)
y_train = np.delete(y_train, train_idx5)


# initializing learner and loading a start dataset for each of them
learner1 = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=entropy_sampling,
    X_training=x_pool1, y_training=y_pool1
    )
learner2 = ActiveLearner(
    estimator=SVC(kernel='rbf', probability=True),
    query_strategy=entropy_sampling,
    X_training=x_pool2, y_training=y_pool2
    )   
learner3 = ActiveLearner(
    estimator=SVC(kernel='poly', probability=True),
    query_strategy=entropy_sampling,
    X_training=x_pool3, y_training=y_pool3
    )  
learner4 = ActiveLearner(
    estimator=GradientBoostingClassifier(),
    query_strategy=entropy_sampling,
    X_training=x_pool4, y_training=y_pool4
    )  
learner5 = ActiveLearner(
    estimator=SVC(kernel='linear', probability=True),
    query_strategy=entropy_sampling,
    X_training=x_pool5, y_training=y_pool5
    )  

# Append all learners into list    
learner_list.append(learner1)
learner_list.append(learner2)
learner_list.append(learner3)
learner_list.append(learner4)
learner_list.append(learner5)

# assembling the committee
committee = Committee(learner_list=learner_list, query_strategy=max_disagreement_sampling)

# Training committee + performance printing

performance_history = [unqueried_score]
performance_f1 = [unqueried_f1]
# query by committee
n_queries = 400                # SET NUMBER OF QUERIES

for idx in range(n_queries):
    query_idx, query_instance = committee.query(x_train)
    committee.teach(
        X=x_train[query_idx].reshape(1, -1),
        y=y_train[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(temp, temp_y))
    dummy_pred = committee.predict(temp)
    performance_f1.append(f1_score(dummy_pred, temp_y, average='macro'))
    if idx % 10 == 0: 
        print('Current iteration:', idx, 'Current F1 (Macro):', f1_score(dummy_pred, temp_y, average='macro'))
    # remove queried instance from pool
    x_train = np.delete(x_train, query_idx, axis=0)
    y_train = np.delete(y_train, query_idx)
    
  

# Performance plots
fig, ax = plt.subplots(figsize=(5, 3), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

plt.show()

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(5, 3), dpi=130)

ax.plot(performance_f1)
ax.scatter(range(len(performance_f1)), performance_f1, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification f1')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification F1')

plt.show()


# Better save than sorry
import pickle

with open('active_learner-level2.pkl', 'wb') as file:
    pickle.dump(committee, file)
    
    
    
#### OPEN NEW COMMITTEE FOR VALIDATION?
# If you want to open another version of a saved committee:
#import pickle
#with open('active_learner-level2.pkl', 'rb') as file:
#    committee = pickle.load(file)

# Classifier uncertainty of unlabelled posts (to collect waves of posts to annotate)

from modAL.uncertainty import classifier_uncertainty

# Unlabbeled dataframe:
df_unl = pd.read_excel('') # Unlabelled data

sentences2 = []
for row in df_unl.itertuples():
    sentences2.append(row.content)

sentence_embeddings2 = model.encode(sentences2)

x_unl = pd.DataFrame(sentence_embeddings2)
predictions = committee.predict(x_unl) 
certainty = classifier_uncertainty(committee, x_unl)

df_unl['uncertainty'] = certainty
df_unl['prediction'] = predictions


uncertain_posts = result
for row in uncertain_posts.itertuples():
    if row.uncertainty < 0.3:   # PICK THRESHOLD
        uncertain_posts = uncertain_posts.drop(i)
    else:
        continue
        
     
    
####### VALIDATION ########

validation = pd.read_excel('')      # Fill in validation dataset
v_sentences = []
for row in validation.itertuples():
    v_sentences.append(row.content)
v_labels = validation.label  


word_embedding_model = models.Transformer("") # Fill in RobBERT model to use for embeddings

# Apply mean pooling or take CLS vector to get one fixed sized sentence vector (SAME AS COMMITTEE TRAINING)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

v_sentence_embeddings = model.encode(v_sentences)
v_emd = pd.DataFrame(v_sentence_embeddings)


predictions2 = committee.predict(v_emd)

# Macro scores across arguments
print('F1 on validation set:', f1_score(v_labels,predictions2, average='macro'))
print('Precision on validation set:', precision_score(v_labels,predictions2, average='macro'))
print('Recall on validation set:', recall_score(v_labels,predictions2, average='macro'))

# Metrics per argument
print('Recall per class:', recall_score(v_labels,predictions2, average=None))
print('Prec per class:', precision_score(v_labels,predictions2, average=None))
print('F1 per class:', f1_score(v_labels,predictions2, average=None))

# Quick plot of confusion matrix
cm = confusion_matrix(predictions2, v_labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  
ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels'); 
ax.set_title('Confusion Matrix'); 