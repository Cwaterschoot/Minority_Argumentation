import numpy as np 
import pandas as pd 
import re
import random
import math
from tqdm.notebook import tqdm
from collections import Counter


from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt 

import wordninja
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

import colibricore
TMPDIR = "tmp/"
import string 
import openpyxl
import xlsxwriter



# Load data
data = pd.read_csv('') # Csv with texts

data_labels = pd.read_csv('') # Csv with labels
data_labels = list(data_labels['train_labels'])
data['labels'] = data_labels


# Stopword filtering
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('dutch'))  

# Some additions to the filter
stop_words.add("amp")
stop_words.add('wel')
stop_words.add('https')
stop_words.add('www')
stop_words.add('gaat')
stop_words.add('gaan')
stop_words.add('wij')
stops = list(stop_words)

lowers = []
for row in data.itertuples():
    strin = row.train_texts.lower()
    strin = strin.translate(str.maketrans('', '', string.punctuation))
    lowers.append(strin)
data['text'] = lowers
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

def flatten_list(l):
    return [x for y in l for x in y]

# 0 = Off-topic
# 11= impact
# 12 = attribution
# 13 = trend
# 14 = no consensus
# 15= bad science
# 16 = conspiracy
#21 = Pro (ACC)

args = [0,11,12,13,14,15,16,21] #


# ONE BIG LOOP THROUGH ARGUMENTS:
for p in args:
    corpus_0 = data[data['labels'] == p]['text'] 
    corpus_0= "\n".join(corpus_0)
    corpus_0 = corpus_0.lower()
    fileloc = 'CC_%s.txt' % (p)
    print(fileloc)
    corpusfile_plaintext = TMPDIR + fileloc

    with open(corpusfile_plaintext,'w',encoding='utf-8') as f:
        f.write(corpus_0)

    classfile = TMPDIR + "cc_0.colibri.cls"
    #Instantiate class encoder
    classencoder = colibricore.ClassEncoder()

    #Build classes
    classencoder.build(corpusfile_plaintext)

    #Save class file
    classencoder.save(classfile)

    print("Encoded arg", i, len(classencoder), " classes, well done!")

    corpusfile = TMPDIR + "CC_0.colibri.dat" #this will be the encoded corpus file
    classencoder.encodefile(corpusfile_plaintext, corpusfile)

    #Load class decoder from the classfile we just made
    classdecoder = colibricore.ClassDecoder(classfile)

    #Decode corpus data
    decoded = classdecoder.decodefile(corpusfile)

    options = colibricore.PatternModelOptions(mintokens=1,minlength=2,maxlength=8)

    #Instantiate an empty unindexed model 
    model = colibricore.UnindexedPatternModel()

    #Train it on our corpus file (class-encoded data, not plain text)
    model.train(corpusfile, options)

    print("Found " , len(model), " patterns:")
    l = []
    #Let's see what patterns are in our model (the order will be 'random')
    for pattern in model:
        #print(pattern.tostring(classdecoder), end=" | ")
        l.append(pattern.tostring(classdecoder))

    patterns = []
    counts = []
    for pattern, count in model.items():
        #print(pattern.tostring(classdecoder), count)
        patterns.append(pattern.tostring(classdecoder))
        counts.append(count)

    df = pd.DataFrame(patterns)
    df['freq'] = counts


    df = df.rename(columns={0: 'patterns', 'freq': 'freq'})
    long_list = []
    for row in df.itertuples():
        freq = row.freq
        for i in range(0,freq):
        #print(i)
            long_list.append(row.patterns)
    #long_list


###############
    
    if p == 0:
        df_0 = df
        l_0 = long_list
        fl = flatten_list([l_14, l_13, l_21, l_16, l_15, l_11, l_12])
    elif p == 11:
        df_11 = df
        l_11 = long_list
        fl = flatten_list([l_14, l_13, l_0, l_16, l_15, l_21, l_12])
    elif p == 12:
        df_12 = df
        l_12 = long_list
        fl = flatten_list([l_14, l_13, l_0, l_16, l_15, l_11, l_21])
    elif p == 13:
        df_13 = df
        l_13 = long_list
        fl = flatten_list([l_14, l_21, l_0, l_16, l_15, l_11, l_12])
    elif p == 14:
        df_14 = df
        l_14 = long_list
        fl = flatten_list([l_21, l_13, l_0, l_16, l_15, l_11, l_12])
    elif p == 15:
        df_15 = df
        l_15 = long_list
        fl = flatten_list([l_14, l_13, l_0, l_16, l_21, l_11, l_12])
    elif p == 16:
        df_16= df
        l_16 = long_list
        fl = flatten_list([l_14, l_13, l_0, l_21, l_15, l_11, l_12])
    else:
        df_21 = df
        l_21 = long_list
        fl = flatten_list([l_14, l_13, l_0, l_16, l_15, l_11, l_12])
        
        
    doc1 = long_list
    doc2 = fl
    
    doc1_counts = Counter(doc1)
    doc1_freq = {
        x: doc1_counts[x]/len(doc1)
        for x in doc1_counts
    }
    
    doc2_counts = Counter(doc2)
    doc2_freq = {
         x: doc2_counts[x]/len(doc2)
         for x in doc2_counts
    }


    words = []
    ll = []
    words_unique = []
    for x in doc1_freq:
        if x not in doc2_freq:
        #print('word not found')
            freq2 = 0
            words_unique.append(x)
        else:
            freq2 = doc2_freq[x]
        e1 = len(l_0) * (doc1_freq[x] + freq2) / (len(fl) + len(l_0))
        e2 = len(fl) * (doc1_freq[x] + freq2) / (len(fl) + len(l_0))
        if x not in doc2_freq:
            logli = (2 * ((doc1_freq[x] * math.log(doc1_freq[x] / e1)) + (freq2 * math.log(1))))
        else:
            logli = (2 * ((doc1_freq[x] * math.log(doc1_freq[x] / e1)) + (freq2 * math.log(freq2 / e2))))
        words.append(x)
        ll.append(logli)

        
    df_ll = pd.DataFrame(words)
    df_unique = pd.DataFrame(words_unique)
    df_ll['LL'] = ll
    df_ll = df_ll.rename(columns={0: 'patterns', 'LL': 'LL'})

    if p == 0:
        df_ll = df_ll.merge(df_0, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_0= df_ll  
    elif p == 11:
        df_ll = df_ll.merge(df_11, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_11= df_ll  
    elif p == 12:
        df_ll = df_ll.merge(df_12, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_12= df_ll  
    elif p == 13:
        df_ll = df_ll.merge(df_13, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_13= df_ll  
    elif p == 14:
        df_ll = df_ll.merge(df_14, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_14= df_ll  
    elif p == 15:
        df_ll = df_ll.merge(df_15, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_15= df_ll  
    elif p == 16:
        df_ll = df_ll.merge(df_16, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_16= df_ll  
    else:
        df_ll = df_ll.merge(df_21, on='patterns') 
        df_ll = df_ll.sort_values(by=['LL'], ascending=False)
        df_ll = df_ll.drop('freq', axis=1)
        #df_ll = df_ll.rename(columns={'patterns': 'ngram', 'LL': 'freq'})
        ll_21= df_ll  
print('All calculated and merged')



dflist= [ll_0, ll_11, ll_12, ll_13, ll_14, ll_15, ll_16, ll_21]
args = ['No_arg','Impact','Attribution','Trend','No_consensus','Bad_science','Conspiracy','AGW']
Excelwriter = pd.ExcelWriter("patterns.xlsx",engine="xlsxwriter")

for i, df in enumerate (dflist):
    argument = args[i]
    df.to_excel(Excelwriter, sheet_name="Sheet" +argument ,index=False)
Excelwriter.save()

print('All saved!')