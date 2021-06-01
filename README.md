# Data-Economy

This repository is dedicated to the project "Valuing the Data Economy: A Labor Costs Approach using Unsupervised Machine Learning". 

# Pre-Processing Steps

## 1. Unpack the Burning Glass Database

The Burning Glass database consists of more than 200 million job postings from 2010 to 2019. The structured databases are contained in several ``zip`` files, while the unstructured data is contained in XML files. 

## Doc2Vec Model Training and Evaluation

To proxy time-use factors, we need to come up with an estimate for the similarity between occupations based on their relationship with data-intensive responsibilities. One way of going about this estimation is by comparing the full set of occupations with known data-intensive occupations using only the language contained in the job posting text. There are a variety of ways to compute document similarities, e.g. TF-IDF, Jaccard Distance, etc., but these bag-of-words methods may miss important semantic similarities between job postings. To this end, we use Doc2Vec to perform our similarity task.

In summary, Doc2vec builds on the Word2Vec algorithm that uses word "contexts" to train a shallow neural network. There are two approaches, continuous bag-of-words and skip-gram. Each approach trains the weights of the neural network to either predict a word's context (skip-gram) or using a word's context to predict itself (continuous bag-of-words). In either case, the trained weights of the neural network correspond to the vector representations of each word in the vocabulary. Doc2Vec builds on this approach by adding a "document tag" that encodes document-level attributes into the Word2Vec model. As a consequence, document vectors are trained alongside word vectors, resulting in a dense vector representation for each document tag.

To dampen the effects of job posting outliers, I train the Doc2Vec model using ONET categories as the document tags. As a consequence, the model trains a vector representation for each ONET category rather than each document. After experimenting with different parameter combinations, I find training the model using the CBOW algorithm, with 1000 dimensions, 15 epochs, a window size of 3 words, and a hierarchical softmax classifier yields the best results. 



Due to limits on our computational resources, I limit the size of the training sample and elect to train a Doc2Vec model for each year in the data. To construct the annual training sample, I start by randomly choosing a week of job postings for a given month. I repeat this sampling process for each month so each month is represented by a weekly job posting file. In other words, our annual training sample is a composite of 12 weekly job postings files. 

After the monthly representatives are sampled and combined into an annual dataset, we split the annual dataset into a training and test sample. As before, our computational resources are limited, and therefore, we randomly sample 1 million job postings from the annual dataset to construct the training sample. We note that for each year we ensure the training and test split is such that the size of the test dataset is at least two-thirds of the training dataset. This is done to ensure an adequate sample remains for validating the trained Doc2Vec model.

After training the Doc2Vec model, I evaluate the model's performance using two common approaches. For each validation procedure, we estimate the performance metric by randomly sampling 50,000 observations with replacement from the test dataset across 5 iterations. Our first metric uses a nearest neighbors criterion to judge the accuracy of the model. For each job posting in the bootstrapped sample, we compute the cosine distance between the inferred job posting's document vector and the representative ONET category embeddings trained by the model. Once computed, we compute a binary metric based on whether the job posting's ONET category is in the 10 nearest ONET category embeddings. We estimate the performance metric by summing up these binary metrics and dividing by the number of job postings in the sample.

Our second approach uses a similarity triplet method. In this approach, we construct a triplet consisting of three job postings. Two of the three job postings are chosen to be in the same ONET category, and the final posting is chosen at random from a different ONET category. To compute the accuracy metric, we assign a binary score to each triplet. The score is equal to one when the cosine distance between the two job postings in the same ONET category is less than the distance between these postings and the job posting from a separate ONET category. The aggregate performance metric is then the sum of all binary scores divided by the total number of triplets constructed from the sample. 

### The Code and Relevant Output





- The code [doc2vec_training_test.py]("doc2vec/doc2vec_training_test.py")
- The size of the training/test datasets is in [train_file.csv]("Data/train_file.csv")
- A feature file containing vocabulary size and the number of trained embeddings is in [feature_file.csv]("Data Economy/Data/feature_file.csv")
- A performance file containing the performance metrics is in [performance_file.csv]("Data/performance_file.csv")

## 2. Task Parsing
One issue with using all visible text from the scraped job postings to train the LDA model is that other job related information may be comingled with the task data. In an effort to reduce commingled data, we combine two natural language processing techniques to extract task information from job postings. The first technique, known as <i>sentence boundary disambiguation</i>, is a technique used to parse sentences within a document by detecting the beginning and end of a sentence. The second technique is a part-of-speech tagging algorithm that classifies the part-of-speech, e.g. noun, verb, adverb, a word token belongs to in a sentence. 

The code [get_tasks.py](get_tasks.py) 
