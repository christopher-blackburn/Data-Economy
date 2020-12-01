# ldaTrain.py 

''' 
----------------------------------------------------------------
Description:
------------
This file runs the topic modeling procedure
----------------------------------------------------------------
'''

import os
import glob
import nltk
import re
import numpy as np 
import pandas as pd 
from pprint import pprint 
import nltk
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# Import spacy for lemmatization (however, unable to have spacy install correctly)
import spacy

# Gensim
import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt 

# Enable logging for gensim -- optional
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



'''
----------------------------------------------------------------
Prepare stopwords
----------------------------------------------------------------
'''
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['pron'])

'''
----------------------------------------------------------------
Import the text data 
----------------------------------------------------------------
'''

# Load the job description data
jobDF = pd.read_csv('/Users/cblackburn/Dropbox/Data Economy/Data/job_description_data.csv',header=None)
print(jobDF.shape)

# Create a function that opens a file and returns the text of the data
def openText(x):
    try:
        os.chdir('/Users/cblackburn/Dropbox/Data Economy/Data')
        f = open(x + 'task.txt','r')
        text_data =  f.readlines()
        if len(text_data) >0:
            return text_data
        else:
            return ''
    except FileNotFoundError:
        return ''
    f.close()


# Get the task data
jobDF['tasks'] = jobDF[4].apply(lambda x: openText(x))

# Drop missing tasks
jobDF = jobDF[jobDF['tasks'] != '']

# Extract the text content
text_data = list(jobDF['tasks'])

'''
----------------------------------------------------------------
Prepare stopwords
----------------------------------------------------------------
'''
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['pron'])

'''
----------------------------------------------------------------
Tokenize words the clean up the text data 
----------------------------------------------------------------
'''
print('Cleaning and tokenizing documents...')
# Remove new line characters
def remove_newline(str):
	return str.replace('\n','')

def remove_intersection(str):
	int_str = ' Hiring Lab Career Advice Browse Jobs Browse Companies Salaries Find Certifications Employer Events Work at Indeed Countries About Help Center Do Not Sell My Personal Information Privacy Center'
	return str.replace(int_str,'')

# Loop through the text data and merge into a single document for each
new_text_data = [' '.join(list(map(remove_intersection,list(map(remove_newline,doc))))) for doc in text_data] 

# Tokenize the documents
def doc_to_words(docs):
	for doc in docs:
		yield(gensim.utils.simple_preprocess(doc,deacc=True))

# Construct a list of word lists
data_words = list(doc_to_words(new_text_data))
data_words_len = [len(doc) for doc in data_words]


# Convert the data words into a pandas dataframe
data_wordsDF = pd.DataFrame(new_text_data,columns=['task'])
data_wordsDF['doc_len'] = data_words_len

# Drop missing tasks 
data_wordsDF = data_wordsDF[data_wordsDF['doc_len'] > 0]

# Winsorize the data 
#lower = data_wordsDF['doc_len'].quantile(0.01)
#upper = data_wordsDF['doc_len'].quantile(0.99)
#data_wordsDF = data_wordsDF[(data_wordsDF['doc_len'] > lower) & (data_wordsDF['doc_len'] < upper)]


# Extract the windsorized data
task_data = list(data_wordsDF['task'])
data_words = list(doc_to_words(task_data))


data_words_net = data_words

'''
----------------------------------------------------------------
Creating Bigram and Trigram Models
----------------------------------------------------------------
This section seems optional but its good practice. There are a
few important things to note:

1. When building the models, the user sets two parameters:
	a. min_count
	b. threshold

2. These parameters could be varied to improve the model 
3. Higher threshold implies fewer phrases
'''

print('Creating bigram and trigram models...')

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words_net,min_count=5,threshold=100)
#trigram = gensim.models.Phrases(bigram[data_words_net],threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)

'''
----------------------------------------------------------------
Remove stopwords, make bigrams, and lemmatize
----------------------------------------------------------------
'''
print('Removing stopwords, making bigrams, and lemmatizing...')

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove stopwords
data_words_nostops = remove_stopwords(data_words_net)

# Form bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component for efficiency
nlp = spacy.load('en',disable=['parser','ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams,allowed_postags=['NOUN','ADJ','VERB','ADV'])


# Replace the word 'datum' with 'data' in the lemmatized version
data_lemmatized_net = []
for task in data_lemmatized:
	task_desc = []
	for word in task:
		if word == 'datum':
			task_desc.append('data')
		else:
			task_desc.append(word)
	data_lemmatized_net.append(task_desc)
    
'''
----------------------------------------------------------------
Create the dictionary and corpus needed for topic modeling
----------------------------------------------------------------
'''
print('Creating dictionary and corpus...')

# Create dictionary
id2word = corpora.Dictionary(data_lemmatized_net)

# Create corpus 
texts = data_lemmatized_net

# Term document frequency
corpus = [id2word.doc2bow(text) for text in texts]


'''
----------------------------------------------------------------
Extracting the data part of a job title
----------------------------------------------------------------
'''

# Create a function to extract the job title
def getTitle(x):
    return x[0:x.find('-')]
jobDF['jobTitle'] = jobDF[0].apply(lambda x: getTitle(x))

# Create a function that returns a 1 if data is in the title
def checkData(x):
    token = word_tokenize(x)
    counter = 0
    for word in token:
        word = word.lower()
        if word == 'data':
            counter += 1
    if counter > 0:
        return 1
    else:
        return 0
    
# Create a function that returns a 1 if databases is in the title
def checkDatabase(x):
    token = word_tokenize(x)
    counter = 0
    for word in token:
        word = word.lower()
        if word == 'database' or word == 'databases':
            counter += 1
    if counter > 0:
        return 1
    else:
        return 0
    
jobDF['dataTitle'] = jobDF['jobTitle'].apply(lambda x: checkData(x))
jobDF['databaseTitle'] = jobDF['jobTitle'].apply(lambda x: checkDatabase(x))
jobDF['dataJob'] = jobDF['dataTitle'] + jobDF['databaseTitle']


# Path to the mallet dataset
mallet_path = '/Users/cblackburn/Downloads/mallet-2.0.8/bin/mallet'

def modelTask(k,jobDF):
    print('Estimating LDA model for k={}...'.format(k))
    
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=k,id2word=id2word,random_seed=42,alpha=50)
    
    print('Extracting topic proportions...')
    # Extract the topic proportions
    doc_topics = list(ldamallet.load_document_topics())
    
    # Loop over the vectors and extract the proportions
    doc_topics_list = []
    for doc in doc_topics:
        doc_top = []
        for topic in doc:
            doc_top.append(topic[1])
        doc_topics_list.append(doc_top)

    # Load the documents to a pandas dataframe
    doc_topics_df = pd.DataFrame(doc_topics_list,index=jobDF.index)

    # Create a new column
    topic_nums = ['topic_{}'.format(x) for x in range(0,k)]

    # Append the topic proportions to the job description dataset
    jobDF[topic_nums] = doc_topics_df
    
    print('Initiating machine learning module...')
    
    # Training data split
    train_set,test_set = train_test_split(jobDF,test_size=0.2,random_state=42)

    # The input features and target
    X_train,y_train = train_set[topic_nums],train_set['dataJob']
    X_test,y_test = test_set[topic_nums],test_set['dataJob']
    
    # Stochastic gradient descent
    sgd_clf = SGDClassifier(random_state=1)
    
    # Random Forest Classifier
    forest_clf = RandomForestClassifier(random_state=1)
    
    # Support Vector Machine
    svm_clf = SVC(random_state=1)
    
    # Multilayer perceptron (NN) Classifier
    mlp_clf = MLPClassifier(random_state = 42)
    
    # Cross validation predictions with SGD
    y_train_pred = cross_val_predict(sgd_clf,X_train,y_train,cv=3)
    
    # Cross validation predictions with RF
    y_probas_forest = cross_val_predict(forest_clf,X_train,y_train,cv=3,method='predict_proba')
    
    # Cross validation with SVM
    y_scores_svm = cross_val_predict(svm_clf,X_train,y_train,cv=3,method='decision_function')

    # Probability of a positive class
    y_scores_forest = y_probas_forest[:,1]
    
    # Probabiliy of a Positive Class
    y_probas_mlp = cross_val_predict(mlp_clf,X_train,y_train,cv=3,method='predict_proba')
    y_mlp_scores = y_probas_mlp[:,1]
    
    # Evaluating the precision versus recall tradeoff
    y_sgd_scores = cross_val_predict(sgd_clf,X_train,y_train,cv=3,method='decision_function')
    
    # Compute the precision of each classifier
    svm_precision = precision_score(y_train,cross_val_predict(svm_clf,X_train,y_train,cv=3))
    sgd_precision = precision_score(y_train,cross_val_predict(sgd_clf,X_train,y_train,cv=3))
    frs_precision = precision_score(y_train,cross_val_predict(forest_clf,X_train,y_train,cv=3))
    mlp_precision = precision_score(y_train,cross_val_predict(mlp_clf,X_train,y_train,cv=3))
    

    # Compute recall
    svm_recall = recall_score(y_train,cross_val_predict(svm_clf,X_train,y_train,cv=3))
    sgd_recall = recall_score(y_train,cross_val_predict(sgd_clf,X_train,y_train,cv=3))
    frs_recall = recall_score(y_train,cross_val_predict(forest_clf,X_train,y_train,cv=3))
    mlp_recall = recall_score(y_train,cross_val_predict(mlp_clf,X_train,y_train,cv=3))
    
    # ROC curve for SGD
    fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train,y_sgd_scores)
    
    # ROC Curve for RF
    fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train,y_scores_forest)
    
    # ROC Curve for SVM
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train,y_scores_svm)
    
    # Compute area under the curve
    auc_sgd = roc_auc_score(y_train,y_sgd_scores)
    auc_frs = roc_auc_score(y_train,y_scores_forest)
    auc_svm = roc_auc_score(y_train,y_scores_svm)
    
    # Drop the topic columns from the data frame
    jobDF = jobDF.drop(columns=topic_nums)
    
    print('Successfully modeled tasks for k={}'.format(k))
    
    return [fpr_sgd,tpr_sgd,fpr_forest,tpr_forest,fpr_svm,tpr_svm,auc_sgd,auc_frs,auc_svm,
           svm_precision,sgd_precision,frs_precision,mlp_precision,svm_recall,sgd_recall,frs_recall,mlp_recall]


# Create index values for each of the relevant components
fpr_sgd_index = 0
tpr_sgd_index = 1
fpr_forest_index = 2
tpr_forest_index = 3
fpr_svm_index = 4
tpr_svm_index = 5
auc_sgd = 6
auc_frs = 7
auc_svm = 8

# Topic number list
k_list = [x*10 for x in range(1,26)]

results = []
for k in k_list:
    results.append(modelTask(k,jobDF))

 '''
-----------------------------------------------------------------
Hyperparameter tuning for the random forest classifier
-----------------------------------------------------------------
For the random forest classifier, we are going to adjusting 
the values for the following hyperparameters:

1. Number of Trees in the Forest (n_estimators)
2. Number of Features considered splitting at each node (max_features)
3. max_depth = max number of levels in each decision tree
4. min_samples_split = min number of data points placed in a node before node is split
5. min_samples_leaf = min number of data points allowed in a leaf node
6. bootstrap = method for sampling data points

# Print the hyperparameters that are in use for the random forest classifier
print(forest_clf.get_params())
'''

print('Estimating LDA Model...')

# Run the LDA model with the optimal number of topics
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=frs_peak_k,id2word=id2word,random_seed=42,alpha=50)
    
print('Extracting topic proportions...')
# Extract the topic proportions
doc_topics = list(ldamallet.load_document_topics())
    
# Loop over the vectors and extract the proportions
doc_topics_list = []
for doc in doc_topics:
    doc_top = []
    for topic in doc:
        doc_top.append(topic[1])
    doc_topics_list.append(doc_top)

# Load the documents to a pandas dataframe
doc_topics_df = pd.DataFrame(doc_topics_list,index=jobDF.index)

# Create a new column
topic_nums = ['topic_{}'.format(x) for x in range(0,frs_peak_k)]

# Append the topic proportions to the job description dataset
jobDF[topic_nums] = doc_topics_df

print('Initiating machine learning module...')

# Training data split
train_set,test_set = train_test_split(jobDF,test_size=0.2,random_state=42)

# The input features and target
X_train,y_train = train_set[topic_nums],train_set['dataJob']
X_test,y_test = test_set[topic_nums],test_set['dataJob']


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

frs_clf_random = RandomizedSearchCV(estimator = forest_clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

print('Initiating random grid search...')
frs_clf_random.fit(X_train,y_train)

print('Grid search complete...')

from sklearn.neural_network import MLPClassifier


# Stochastic gradient descent
sgd_clf = SGDClassifier(random_state=1)

# Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=1)

# Support Vector Machine
svm_clf = SVC(random_state=1)

# Multilayer perceptron (NN) Classifier
mlp_clf = MLPClassifier(random_state = 42)


# Cross validation predictions with SGD
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train,cv=3)

# Cross validation predictions with RF
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train,cv=3,method='predict_proba')

# Cross validation with SVM
y_scores_svm = cross_val_predict(svm_clf,X_train,y_train,cv=3,method='decision_function')

# Probability of a positive class
y_scores_forest = y_probas_forest[:,1]

# Evaluating the precision versus recall tradeoff
y_sgd_scores = cross_val_predict(sgd_clf,X_train,y_train,cv=3,method='decision_function')

# Evaluate the preceision recall tradeoff
y_probas_mlp = cross_val_predict(mlp_clf,X_train,y_train,cv=3,method='predict_proba')
y_mlp_scores = y_probas_mlp[:,1]


# ROC curve for SGD
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train,y_sgd_scores)

# ROC Curve for RF
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train,y_scores_forest)

# ROC Curve for SVM
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train,y_scores_svm)

# ROC Curve for MLP
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_train,y_mlp_scores)

# Compute area under the curve
auc_sgd = roc_auc_score(y_train,y_sgd_scores)
auc_frs = roc_auc_score(y_train,y_scores_forest)
auc_svm = roc_auc_score(y_train,y_scores_svm)
auc_mlp = roc_auc_score(y_train,y_mlp_scores)

'''
------------------------------------------------------------------------------------------------
Note: Random Forests and the Multilayer perceptron perform very well compared to the other 
      algorithms. In this section, we explore some of the characteristics of the predictions.
      
      We start with the default hyperparameters. In the next section, we will fine tune the
      hyperparameters. 
------------------------------------------------------------------------------------------------
'''
# Compute the precision of each classifier
frs_precision = precision_score(y_train,cross_val_predict(forest_clf,X_train,y_train,cv=3))
mlp_precision = precision_score(y_train,cross_val_predict(mlp_clf,X_train,y_train,cv=3))

# Compute recall
frs_recall = recall_score(y_train,cross_val_predict(forest_clf,X_train,y_train,cv=3))
mlp_recall = recall_score(y_train,cross_val_predict(mlp_clf,X_train,y_train,cv=3))

print(frs_precision,mlp_precision)
print(frs_recall,mlp_recall)

'''
-----------------------------------------------------------------
Tuning hyperparameters for random forest (continued)
-----------------------------------------------------------------
'''


# Get the best estimator
frs_clf_best = frs_clf_random.best_estimator_

# Evaluate the performance of the best estimator
frs_best_precision = precision_score(y_train,cross_val_predict(frs_clf_best,X_train,y_train,cv=3))
frs_best_recall = recall_score(y_train,cross_val_predict(frs_clf_best,X_train,y_train,cv=3))

print(frs_best_precision,frs_best_recall,frs_clf_best)

 
'''
-----------------------------------------------------------------
Tuning the hyperparameters for the multilayer perceptron 
-----------------------------------------------------------------
Adjusting the False Positives
'''
print('Estimating LDA Model...')

# Run the LDA model with the optimal number of topics
ldamallet_mlp = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=mlp_peak_k,id2word=id2word,random_seed=42,alpha=50)
    
print('Extracting topic proportions...')
# Extract the topic proportions
doc_topics_mlp = list(ldamallet_mlp.load_document_topics())
    
# Loop over the vectors and extract the proportions
doc_topics_list_mlp = []
for doc in doc_topics_mlp:
    doc_top = []
    for topic in doc:
        doc_top.append(topic[1])
    doc_topics_list_mlp.append(doc_top)

# Load the documents to a pandas dataframe
doc_topicsmlp_df = pd.DataFrame(doc_topics_list_mlp,index=jobDF.index)

# Create a new column
topic_nums_mlp = ['topic_{}'.format(x) for x in range(0,mlp_peak_k)]

# Append the topic proportions to the job description dataset
jobDF[topic_nums_mlp] = doc_topicsmlp_df

print('Initiating machine learning module...')

# Training data split
#train_set,test_set = train_test_split(jobDF,test_size=0.2,random_state=42)

# The input features and target
X_train,y_train = train_set[topic_nums_mlp],train_set['dataJob']
X_test,y_test = test_set[topic_nums_mlp],test_set['dataJob']






# Number of hidden layers 
hidden_layer_sizes = [(50,),(100,),(150,),(200,)]

# Initial learning rate
learning_rate_init = [0.00001,0.0001,0.001,0.01]

# Activation function
activation = ['tanh','relu']

# L2 Regularization parameter
alpha = [0.00001,0.0001,0.001, 0.01,0.1]

# Solver 
solver = ['sgd','adam']

# Learning rate
learning_rate = ['constant','adaptive']

# Create the random grid
random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'learning_rate_init': learning_rate_init,
               'activation': activation,
               'alpha': alpha,
               'solver': solver,
               'learning_rate': learning_rate}

mlp_clf_random = RandomizedSearchCV(estimator = mlp_clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

print('Initiating random grid search...')
mlp_clf_random.fit(X_train,y_train)

# Get the best estimator
mlp_clf_best = mlp_clf_random.best_estimator_

# Evaluate the performance of the best estimator
mlp_best_precision = precision_score(y_train,cross_val_predict(mlp_clf_best,X_train,y_train,cv=3))
mlp_best_recall = recall_score(y_train,cross_val_predict(mlp_clf_best,X_train,y_train,cv=3))

print(mlp_best_precision,mlp_best_recall,mlp_clf_best)







