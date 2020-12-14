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
from xml.dom import minidom 

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

print('Completed!')


print('Estimating LDA Model for Random Forests...')

# Number of topics for the Random Forests model
frs_peak_k = 50

# Path to the mallet dataset
mallet_path = '/Users/cblackburn/Downloads/mallet-2.0.8/bin/mallet'

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

# Training data split
train_set,test_set = train_test_split(jobDF,test_size=0.2,random_state=42)

# The input features and target
X_train,y_train = train_set[topic_nums],train_set['dataJob']
X_test,y_test = test_set[topic_nums],test_set['dataJob']

print('Estimating Random Forests Classifier...')

# Random Forest Classifier
frs_clf = RandomForestClassifier(random_state=1,bootstrap=False,max_depth=50,n_estimators=1000)

frs_clf_best = frs_clf.fit(X_train,y_train)

print('Completed')


# getBurningJobs.py

'''
This script accesses the jobs within the Burnign Glass XML files
'''




# Set the working directory to the sample XML files
os.chdir('/Users/cblackburn')

# Parse the XML document 
mydoc = minidom.parse('US_XML_AddFeed_20100101_20100107.xml')

# Extract each "Job" element
items = mydoc.getElementsByTagName('Job')



# Next, we can iterate within the each of these to extract the relevant information.
job_info_array = []
for node in items:
    try:
        # Extract the Job ID
        jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue
    
        # Extract the raw text data from the job
        jobText = node.getElementsByTagName('JobText')[0].firstChild.nodeValue

    except AttributeError:
        # Extract the Job ID
        jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue
        
        # Assign an empty job posting text
        jobText = ''

    # Append the job information to the job information array
    job_info_array.append([jobID,jobText])


# Convert to a pandas DataFrame
job_info_df = pd.DataFrame(job_info_array,columns=['JobID','JobText'])

# Save the data 
os.chdir('/Users/cblackburn/Dropbox/Data Economy/BurnGlassData')
job_info_df.to_csv('burn_glass_test.csv',index=False)
print('Completed')



# Clean the text data
bg_text_data = list(job_info_df['JobText'])

bg_text_clean = []
for text in bg_text_data:
    if type(text) == float:
        bg_text_clean.append('Missing String')
    else:
        clean_text = text.replace('\n','')
        bg_text_clean.append(clean_text.replace('\t',''))
        
print('Processing the data...')
data_words_net = list(doc_to_words(bg_text_clean))
data_words_nostops = remove_stopwords(data_words_net)

data_lemmatized = lemmatization(data_words_nostops)

data_lemmatized_net = []
for task in data_lemmatized:
    task_desc = []
    for word in task:
        if word == 'datum':
            task_desc.append('data')
        else:
            task_desc.append(word)
    data_lemmatized_net.append(task_desc)
    
# Create the corpus 
texts = data_lemmatized_net

corpus = [id2word.doc2bow(text) for text in texts]

bg_results = ldamallet[corpus]

doc_topics_list = []
for doc in bg_results:
    doc_top = []
    for topic in doc:
        doc_top.append(topic[1])
    doc_topics_list.append(doc_top)
    
bgDF = pd.DataFrame(doc_topics_list)

print('Completed')


# Estimate the probabilities
y_probas_frs = frs_clf_best.predict_proba(bgDF)

job_info_df['frs_prob'] = y_probas_frs[:,1]

# Open the data
os.chdir('/Users/cblackburn/Downloads')
main_bg_data = pd.read_csv('Main_2010-01.csv',sep=",",header=0)
main_bg_data.head()

from ftplib import FTP
from os import path

saveDir = '/Users/cblackburn/Downloads/'
expDir = '/Users/cblackburn/Dropbox/Data Economy/Output/'

print('Logging into FTP...')
# Login to the FTP server
ftphost = 'transfer.burning-glass.com'
userID = 'USBuofEconAnalysis'
userPW = 'USBUECON20'
ftp = FTP(ftphost)
ftp.login(user=userID,passwd=userPW)

print('Retrieving file names...')
# Loop through the directories and get the list of filenames
text_data_filenames = []
struct_data_filenames = []
for year in range(2010,2011):
    targetDir = '/Burning Glass Data/Text Data/{}'.format(str(year))
    ftp.cwd(targetDir)
    for filename in ftp.nlst():
        text_data_filenames.append(filename)
    targetDir = '/Burning Glass Data/Structured Data/Main/{}'.format(str(year))
    ftp.cwd(targetDir)
    for filename in ftp.nlst():
        struct_data_filenames=[]
        
        
    
# Loop through the text data files and apply the prediction model
pattern = 'AddFeed\_(.*?)\.zip'
year_list = []
for td_file in text_data_filenames: 
    
    print('Extracting timestamp information...')
    # Regex for finding the year and month of the data
    substring = re.search(pattern,td_file).group(1)
    year = substring[0:4]
    month = substring[4:6]
    day = substring[6:8]
    
    year_list.append(year)
    
    if path.exists(expDir + 'data_jobs_{}{}{}.csv'.format(str(year),str(month),str(day))) == True:
        pass
    else:

        print('Checking for files on local hardrive...')
        # Check if the data is currently downloaded 
        if path.exists(saveDir + td_file) == False:
            print('Retrieving Text File for {}-{}-{} from Server'.format(str(year),str(month),str(day)))
            try:
                targetDir = '/Burning Glass Data/Text Data/{}'.format(str(year))
                ftp.cwd(targetDir)
                with open(saveDir + td_file,'wb') as f:
                    ftp.retrbinary('RETR ' + td_file,f.write)

            except EOFError:

                # Login to the FTP server
                ftp = FTP(ftphost)
                ftp.login(user=userID,passwd=userPW)
                targetDir = '/Burning Glass Data/Text Data/{}'.format(str(year))
                ftp.cwd(targetDir)
                with open(saveDir + td_file,'wb') as f:
                    ftp.retrbinary('RETR ' + td_file,f.write)


        else:
            pass 

        print('Unzipping files if needed...')
        # Check if the file is unzipped 
        unzip_td_file = td_file[:-3] + 'xml'
        if path.exists(saveDir + unzip_td_file) == False:
            with ZipFile(saveDir + td_file,'r') as zip_ref:
                zip_ref.extractall(saveDir.rstrip('/'))
        else:
            pass

        # Check if the monthly data is downloaded
        main_filename = 'Main_{}-{}.zip'.format(str(year),str(month))
        if path.exists(saveDir + main_filename) == False:
            try:
                print('Retrieving Main File for {}-{}'.format(str(year),str(month)))
                targetDir = '/Burning Glass Data/Structured Data/Main/{}'.format(str(year))
                ftp.cwd(targetDir)
                with open(saveDir + main_filename,'wb') as f:
                    ftp.retrbinary('RETR ' + main_filename,f.write)
            except EOFError:
                # Login to the FTP server
                ftp = FTP(ftphost)
                ftp.login(user=userID,passwd=userPW)
                targetDir = '/Burning Glass Data/Structured Data/Main/{}'.format(str(year))
                ftp.cwd(targetDir)
                with open(saveDir + main_filename,'wb') as f:
                    ftp.retrbinary('RETR ' + main_filename,f.write)

        else:
            pass

        print('Unzipping files if needed...')
        # Unzip the monthly file if not already unzipped
        unzip_main_file = main_filename[:-3] + 'txt'
        if path.exists(saveDir + unzip_main_file) == False:
            with ZipFile(saveDir + main_filename,'r') as zip_ref:
                zip_ref.extractall(saveDir.rstrip('/'))
        else:
            pass


        print('Parsing XML document...')
        # Parse the XML document 
        parse_doc = saveDir + td_file[:-3] + 'xml'
        mydoc = minidom.parse(parse_doc)


        # Extract each "Job" element
        items = mydoc.getElementsByTagName('Job')


        print('Extracting job-related information...')
        # Next, we can iterate within the each of these to extract the relevant information.
        job_info_array = []
        for node in items:
            try:
                # Extract the Job ID
                jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue

                # Extract the raw text data from the job
                jobText = node.getElementsByTagName('JobText')[0].firstChild.nodeValue

            except AttributeError:
                # Extract the Job ID
                jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue

                # Assign an empty job posting text
                jobText = ''

            # Append the job information to the job information array
            job_info_array.append([jobID,jobText])


        # Convert to a pandas DataFrame
        job_info_df = pd.DataFrame(job_info_array,columns=['JobID','JobText'])

        # Clean the text data
        bg_text_data = list(job_info_df['JobText'])


        print('Cleaning job text data...')
        bg_text_clean = []
        for text in bg_text_data:
            if type(text) == float:
                bg_text_clean.append('Missing String')
            else:
                clean_text = text.replace('\n','')
                bg_text_clean.append(clean_text.replace('\t',''))

        data_words_net = list(doc_to_words(bg_text_clean))
        data_words_nostops = remove_stopwords(data_words_net)

        print('Lemmatizing job text data...')
        data_lemmatized = lemmatization(data_words_nostops)

        data_lemmatized_net = []
        for task in data_lemmatized:
            task_desc = []
            for word in task:
                if word == 'datum':
                    task_desc.append('data')
                else:
                    task_desc.append(word)
            data_lemmatized_net.append(task_desc)

        # Create the corpus 
        texts = data_lemmatized_net

        corpus = [id2word.doc2bow(text) for text in texts]


        print('Applying topic model to corpus...')
        bg_results = ldamallet[corpus]

        doc_topics_list = []
        for doc in bg_results:
            doc_top = []
            for topic in doc:
                doc_top.append(topic[1])
            doc_topics_list.append(doc_top)

        bgDF = pd.DataFrame(doc_topics_list)


        print('Applying machine learning model to data...')
        # Estimate the probabilities
        y_probas_frs = frs_clf_best.predict_proba(bgDF)
        job_info_df['frs_prob'] = y_probas_frs[:,1]

        # Merge with the main dataset
        job_info_df.columns = ['BGTJobId','JobText','frs_prob']
        job_info_df['BGTJobId'] = job_info_df['BGTJobId'].apply(lambda x: int(x))

        print('Loading main dataset...')
        # Load the main data file 
        main_bg_data = pd.read_csv(unzip_main_file,sep='\t',encoding='ISO-8859-1')

        # Merge with the main BG data
        main_merge = pd.merge(main_bg_data,job_info_df,how='inner',on='BGTJobId',validate='1:1')

        # Get whether data is in the job title
        main_merge['dataJob'] = main_merge['CleanTitle'].apply(lambda x: getDataJob(x))

        # Check if data is in the text
        main_merge['dataText'] = main_merge['JobText'].apply(lambda x: checkText(x))

        # Let's reduce the dataset to what we want
        data_jobs = main_merge[(main_merge['dataJob'] == 1) | (main_merge['frs_prob'] >= 0.4) & (main_merge['dataText'] == 1)]

        # Export the dataset
        
        data_jobs.to_csv(expDir + 'data_jobs_{}{}{}.csv'.format(str(year),str(month),str(day)),index=False)

        # Count the number of times the year appears in the list
        year_counter = year_list.count(year)

        # Remove the files to manage storage
        print('Removing weekly file for {}-{}-{}'.format(str(year),str(month),str(day)))
        os.remove(saveDir + td_file)
        os.remove(saveDir + unzip_td_file)

        files54 = ['2010','2011','2013','2014','2015','2017']
        files53 = ['2012','2016','2018','2019']

        if year_counter == 53 and year in files53:
            print('Removing main files for {}...'.format(str(year)))
            for f in glob(saveDir + 'Main*'):
                os.remove(f)
        elif year_counter == 54 and year in files54:
            print('Removing main files for {}...'.format(str(year)))
            for f in glob(saveDir + 'Main*'):
                os.remove(f)
        
ftp.quit()






