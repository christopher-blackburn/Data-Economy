import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from time import time
import multiprocessing
import re
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.model_selection import train_test_split

import random
import glob
import csv

'''
---------------------------------------------------
Randomly sample data for each year
---------------------------------------------------
Due to limits on our computational resources,
we must limit the size of our training sample.
Given these constraints, we elect to train a
Doc2Vec model for each year in the data. To
construct the annual sample, we start by 
randomly sampling a weekly file of job
postings for a given month. We repeat this
sampling process for each month such that
each month is represented by a weekly file. 
Thus, our annual dataset is a composite
of 12 weekly job posting files.
'''


# Function for retrieving a sample of filenames for a given year
def annual_sample(year):
    # Update the directory information
    textDir = 'E:/Research1/prediction/Burning_glass/Text Data/{}'.format(year)
    mainDir = 'E:/Research1/prediction/Burning_glass/Structured Data/Main/{}'.format(year)

    rel_files = []

    # Randomly sample a week from each month
    for month in range(1, 13):

        if month < 10:

            # Randomly sample the file for a month
            rel_file = random.sample([month_file for month_file
                                      in glob.glob(textDir + '/job_info_{}0{}*.csv'.format(year, month))], 1)[0]

            # Open the main dataset
            job_main = pd.read_csv(mainDir + '/Main_{}-0{}.txt'.format(year, month),
                                   sep='\t', encoding='ISO-8859-1').drop_duplicates()

        else:

            # Randomly sample a file for the month
            rel_file = random.sample([month_file for month_file
                                      in glob.glob(textDir + '/job_info_{}{}*.csv'.format(year, month))], 1)[0]

            # Open the main dataset
            job_main = pd.read_csv(mainDir + '/Main_{}-{}.txt'.format(year, month),
                                   sep='\t', encoding='ISO-8859-1').drop_duplicates()

        # Open the text file as a pandas dataframe
        job_text = pd.read_csv(rel_file.replace('\\', '/'), names=['BGTJobId', 'jobText']).drop_duplicates()

        # Merge the text file and append to the relevant file list
        try:
            rel_files.append(pd.merge(job_text, job_main, on='BGTJobId', how='inner', validate='1:1'))
        except ValueError:
            job_main['BGTJobId'] = job_main['BGTJobId'].astype('object')
            rel_files.append(pd.merge(job_text, job_main, on='BGTJobId', how='inner', validate='1:1'))

    # Concatenate all of the monthly files into a single dataset
    job_data = pd.concat(rel_files).drop_duplicates()

    return job_data


'''
---------------------------------------------------
The training and test splitting process
---------------------------------------------------
After the monthly representatives are sampled
and combined into an annual dataset, we split
the annual dataset into a training and test
sample. As before, our computational resources
are limited, and therefore, we randomly sample
1 million job postings from the annual dataset
to construct the training sample.\footnote{
For each year, we ensure the training and test
split is such that the test dataset is at least 
two-thirds of our annual sample. This is done 
to ensure an adequate sample remains for 
validating the trained Doc2Vec model.
} 

Figure XX shows the training and test split for
each year. 
'''


def train_test(job_data, year):
    '''
    This function returns a split of the dataset
    into a training and test dataset.

    Inputs:
    -------
    job_data : a Pandas DataFrame returned from
               the function annual_sample(year)

    Outputs:
    --------
    train_data : Pandas DataFrame (training data)
    test_data  :  Pandas Dataframes (test data)

    test_data.shape[0],
    train_data.shape[0]


    '''

    # Start by making sure the dataset is large enough for the split
    if job_data.shape[0] > 1200000:

        # Construct test and training sample
        test_size = (1 - 1000000 / job_data.shape[0])
        train_data, test_data = train_test_split(job_data, test_size=test_size)

        return train_data, test_data, train_data.shape[0], test_data.shape[0]

    else:

        print('Re-running annual sampling program', end='...')
        # Re-run the sampling function to acquire more data
        job_data_supp = annual_sample(year)

        # Append the data to the original sample
        job_data = pd.concat([job_data, job_data_supp]).drop_duplicates()

        # Make sure the dataset is large enough for the split
        if job_data.shape[0] > 1200000:

            # Construct test and training sample
            test_size = (1 - 1000000 / job_data.shape[0])
            train_data, test_data = train_test_split(job_data, test_size=test_size)

            return train_data, test_data, train_data.shape[0], test_data.shape[0]

        else:
            print('Re-running annual sampling program', end='...')

            # Re-run the sampling function to acquire more data
            job_data_supp = annual_sample(year)

            # Append the data to the original sample
            job_data = pd.concat([job_data, job_data_supp]).drop_duplicates()

            # Make sure the dataset is large enough for the split
            if job_data.shape[0] > 1200000:

                # Construct test and training sample
                test_size = (1 - 1000000 / job_data.shape[0])
                train_data, test_data = train_test_split(job_data, test_size=test_size)

                return train_data, test_data, train_data.shape[0], test_data.shape[0]

            else:

                print('Re-running annual sampling program', end='...')

                # Re-run the sampling function to acquire more data
                job_data_supp = annual_sample(year)

                # Append the data to the original sample
                job_data = pd.concat([job_data, job_data_supp]).drop_duplicates()

                # Make sure the dataset is large enough for the split
                if job_data.shape[0] > 1200000:

                    # Construct test and training sample
                    train_data = job_data.sample(n=1000000)
                    test_data = job_data.drop(train_data.index)

                    return train_data, test_data, train_data.shape[0], test_data.shape[0]
                else:
                    print('Re-running annual sampling program', end='...')

                    # Re-run the sampling function to acquire more data
                    job_data_supp = annual_sample(year)

                    # Append the data to the original sample
                    job_data = pd.concat([job_data, job_data_supp]).drop_duplicates()

                    # Make sure the dataset is large enough for the split
                    if job_data.shape[0] > 1200000:

                        # Construct test and training sample
                        test_size = (1 - 1000000 / job_data.shape[0])
                        train_data, test_data = train_test_split(job_data, test_size=test_size)

                        return train_data, test_data, train_data.shape[0], test_data.shape[0]

                    else:

                        print('Re-running annual sampling program', end='...')

                        # Re-run the sampling function to acquire more data
                        job_data_supp = annual_sample(year)

                        # Append the data to the original sample
                        job_data = pd.concat([job_data, job_data_supp]).drop_duplicates()

                        # Make sure the dataset is large enough for the split
                        if job_data.shape[0] > 1200000:

                            # Construct test and training sample
                            test_size = (1 - 1000000 / job_data.shape[0])
                            train_data, test_data = train_test_split(job_data, test_size=test_size)

                            return train_data, test_data, train_data.shape[0], test_data.shape[0]

                        else:

                            print('Only {} observations! Consider checking the data'.format(job_data.shape[0]))


'''
---------------------------------------------------
Doc2Vec Model Training
---------------------------------------------------
Using the test dataset, we train a Doc2Vec model 
for each year. After several experiments, we elect
to fit a model using 1,000 dimensional vectors using
a hierarchical softmax classifier. 

On the margin, we did not achieve enough gains from
increasing the number of epochs in the training
runs to justifies the incremental time cost. Thus,
we train each model using 10 epochs. 
'''


def doc2vec_training(train_data, year):
    '''
    This function trains a Doc2Vec model on the training data

    Inputs:
    -------
    train_data : Pandas DataFrame of training data
    year : Int or str specifying the year of the run


    Outputs:
    --------
    model : The fitted doc2vec model
    len(model.wv.vocab) : Length of the vocabulary
    len(model.docvecs) : Number of fitted document vectors

    '''

    # Tag the documents using the ONET codes
    jobtext_corpus = [TaggedDocument(simple_preprocess(posting[0]), [posting[1]])
                      for posting in zip(list(train_data['jobText']), list(train_data['ONET']))]

    # Number of CPU cores used for the training
    num_workers = multiprocessing.cpu_count()

    # Instantiate a Doc2Vec model with the appropriate hyperparameters
    model = Doc2Vec(vector_size=1000, min_count=3, epochs=15, workers=num_workers, dm=0, hs=1, window=3, dbow_words=0)

    # Build the vocabulary
    model.build_vocab(jobtext_corpus)

    # Train the model
    model.train(jobtext_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the model
    model.save('E:/Research1/prediction/Burning_glass/Chris/Word2Vec/w2v{}'.format(year))

    return model, len(model.wv.vocab), len(model.docvecs)


'''
---------------------------------------------------
Training Evaluation
---------------------------------------------------
After training a Doc2Vec model, we evualute the 
model's performance using two common approaches.
Before introducing these approaches, we notethat 
we run 5 iterations for each model to reduce the 
time cost of the evaluation program. In each iteration,
we randomly select 50,000 observations from the 
test dataset to compute our evaluation metrics.

Our first metric uses a nearest neighbors criterion to 
judge the accuracy of the model. For each
job posting in our subsample test dataset, 
we compute the cosine distance between the 
job posting's document vector and the representative
ONET category embeddings trained by the model. 
Subsequently, we check whether the job posting's 
ONET category is in the 10 nearest ONET 
category embeddings. 

Our second approach uses a similarity triplet method. In
this approach, we construct a triplet consisting
of three job postings. Two of the three postings 
are chosen to be in the same ONET category, 
and the final posting is chosen at random from a
different ONET category. To compute the accuracy metric,
we assign a binary score to each triplet. 
The score is equal to one when the cosine
distance between the two similar job postings
is smaller than the distance between these postings
and the random job posting. Otherwise, the triplet 
is assigned a score of zero. 
'''


# Function for sampling the test dataset
def evaluate_model(test_data, doc2vec_model, sample_size=50000):
    # Sample the training data
    test_sample_data = test_data.sample(n=sample_size)

    # Construct tagged documents
    tagged_docs = [TaggedDocument(simple_preprocess(posting[0]), [posting[1]])
                   for posting in zip(list(test_sample_data['jobText']), list(test_sample_data['ONET']))]

    # Construct a pandas dataframe from the tagged documents
    wordsDF = pd.DataFrame([[job.words, job.tags[0]] for job in tagged_docs])

    # Function for checking if the category is in the nearest neighborhood
    def check_neighborhood(job):
        words, tags = job.words, job.tags
        try:
            sims = np.array(doc2vec_model.docvecs.most_similar([doc2vec_model.infer_vector(words)]))
            if float(sims[np.where(sims == tags)[0][0], 1]):
                return 1
        except:
            return 0

    # Function for retrieving the ranks
    def get_neighb_rank(job):
        words, tags = job.words, job.tags
        try:
            sims = np.array(doc2vec_model.docvecs.most_similar([doc2vec_model.infer_vector(words)]))
            return float(np.where(sims == tags)[0][0])
        except:
            return np.nan

            # Compute the nearest neighbor criterion score

    nn_score = np.mean([check_neighborhood(job) for job in tagged_docs])

    # Compute the average rank for those in the top 10
    nn_rank = np.mean([rank for rank in
                       [get_neighb_rank(job) for job in tagged_docs] if np.isnan(rank) == False])

    # Function to compute the triplet score
    def eval_triplet(wordsDF, doc2vec_model):
        run_avg = []
        for soc in list(wordsDF[1].unique()):
            try:
                wordsDF_sample = list(wordsDF[wordsDF[1] == soc].sample(2)[0])
                wordsDF_sample_random = list(wordsDF[wordsDF[1] != soc].sample(1)[0])
                triplet_vectors = np.array([doc2vec_model.infer_vector(wordsDF_sample[0]),
                                            doc2vec_model.infer_vector(wordsDF_sample[1]),
                                            doc2vec_model.infer_vector(wordsDF_sample_random[0])])

                dist_matrix = squareform(pdist(triplet_vectors, 'cosine'))
                if dist_matrix[0, 1] < dist_matrix[0, 2]:
                    run_avg.append(1)
                else:
                    run_avg.append(0)
            except ValueError:
                pass

        return np.mean(run_avg)

    triplet_score = eval_triplet(wordsDF, doc2vec_model)

    return nn_score, nn_rank, triplet_score


'''
---------------------------------------------------
Utility functions for saving data
---------------------------------------------------
In each training iteration, we save several 
important variables.
'''

import csv


# A function to dynamically update the test/train split file
def save_train_features(year, train_data_size, test_data_size):
    train_file = 'E:/Research1/prediction/Burning_glass/Chris/Output/train_file.csv'
    train_file_add = pd.DataFrame([[year, train_data_size, test_data_size]],
                                  columns=['year', 'train_data_size', 'test_data_size'])
    try:
        train_file_data = pd.read_csv(train_file)
        train_file_data = train_file_data.append(train_file_add)
        train_file_data.to_csv(train_file, index=False)

    except FileNotFoundError:
        train_file_add.to_csv(train_file, index=False)


# A function to dynamically update the model features file
def save_model_features(year, vocab_size, wv_size):
    feature_file = 'E:/Research1/prediction/Burning_glass/Chris/Output/feature_file.csv'
    feature_file_add = pd.DataFrame([[year, vocab_size, wv_size]], columns=['year', 'vocab_size', 'wv_size'])

    try:
        feature_file_data = pd.read_csv(feature_file)
        feature_file_data = feature_file_data.append(feature_file_add)
        feature_file_data.to_csv(feature_file, index=False)


    except FileNotFoundError:
        feature_file_add.to_csv(feature_file, index=False)


# A function to dynamically upate the model performance details
def save_model_performance(year, nn_score, nn_rank, triplet_score):
    performance_file = 'E:/Research1/prediction/Burning_glass/Chris/Output/performance_file.csv'
    perf_file_add = pd.DataFrame([[year, nn_score, nn_rank, triplet_score]],
                                 columns=['year', 'nn_score', 'nn_rank', 'triplet_score'])
    try:
        perf_file_data = pd.read_csv(performance_file)
        perf_file_data = perf_file_data.append(perf_file_add)
        perf_file_data.to_csv(performance_file, index=False)

    except FileNotFoundError:
        perf_file_add.to_csv(performance_file, index=False)


'''
---------------------------------------------------
Executing the training and testing program
---------------------------------------------------
'''

for year in range(2014, 2018):

    print('##############################')
    print('####### YEAR {} Starting #####'.format(year))
    print('##############################')

    # 1. Construct a sample of job postings for the year
    print('Constructing annual sample', end='...')
    start = time()
    job_data = annual_sample(year)
    print(
        'Annual sample constructed after {} seconds. Total runtime: {} seconds'.format(time() - start, time() - start))

    # 2. Perform the train/test split
    print('Performing the train/test split', end='...')
    start_ = time()
    train_data, test_data, train_data_size, test_data_size = train_test(job_data, year)
    print(
        'Train/test split completed after {} seconds. Two-thirds requirement satisfied. Total runtime: {} seconds'.format(
            time() - start_, time() - start))

    # 2a. Save the train/test features data
    save_train_features(year, train_data_size, test_data_size)
    print('Saved train/test split information!')

    # 3. Train the Doc2Vec model
    print('Initiating Doc2Vec training')
    start_ = time()
    try:
        model, vocab_size, wv_size = doc2vec_training(train_data, year)
    except TypeError:
        train_data = train_data.dropna(subset=['jobText'])
        model, vocab_size, wv_size = doc2vec_training(train_data, year)
    print('Doc2Vec training completed after {} seconds.'.format(time() - start_), end='...')

    # 3a. Save the model features
    save_model_features(year, vocab_size, wv_size)
    print('Model features saved. Total runtime: {} seconds'.format(time() - start))

    # 4. Evaluate the model's performance
    print('Evaluating model performance', end='...')
    nn_score, nn_rank, triplet_score = evaluate_model(test_data=test_data, doc2vec_model=model, sample_size=50000)
    print('Model results computed: NN = {}, Rank = {}, Triplet = {}'.format(nn_score, nn_rank, triplet_score), end
          ='...')

    # 4a. Save the model performance results
    save_model_performance(year, nn_score, nn_rank, triplet_score)
    print('Model results saved. Total runtime: {} seconds'.format(time() - start))

    print('##############################')
    print('##### YEAR {} Complete #####'.format(year))
    print('##############################')


