# Data-Economy

This repository is dedicated to the project "Valuing the Data Economy: A Labor Costs Approach using Unsupervised Machine Learning". After we received the full Burning Glass Dataset on October 7th, 2020, the methodology changed from using Latent Dirichlet Allocation to Doc2Vec. The description contained in this is README pertains to the Doc2Vec version of the methodology that uses the Burning Glass Database. 

# Introduction

The collection, analysis, and distribution of data is a hallmark of modern economic activity. The rapid proliferation of this so-called "data economy" has instantiated a call to action among academic, political, and public forums to better understand the economic value created by these activities. However, estimating the value of the data economy is complicated by the fact that the quantity of data collected is not readily observed and companies have strong incentives to hoard their most valuable data. Moreover, traditional economic surveys have been slow to keep pace with the rapidly evolving landscape of the data economy, making survey-based approaches for valuation untenable. With these complications, producing an accurate, reliable, and timely estimate for the value of the data economy requires extending traditional measurement approaches to leverage new data sources. 

Confronting this challenge, I introduce a new method for valuing the data economy using online job postings. The method applies an unsupervised machine learning algorithm to online job advertisements to estimate the labor costs of data-related activities. Our method augments the traditional labor costs methodology by proxying time-use factors using only the language contained in job posting text. Using this method, we estimate data-related (nominal) labor costs grew from $100 billion in 2010 to more than $200 billion, representing an average annual growth rate of 9 percent. 

We elect to estimate the value of the data economy using spending on data-related tasks for two reasons. First, other approaches, such as transaction-based approaches, may severely understate spending on data if only a small fraction of the data economy takes place on open markets. A growing concern is related to the amount of data automatically collected from digital devices, such as smartphones and laptops. The widespread diffusion of these devices, along with a concomitant acceleration in digital service offerings, suggests a non-trivial fraction of data may collected outside of standard market transactions. 

Second, the labor costs approach taken in this paper is consistent with current national accounting practices. This is important because current national accounting practices exclude the value of data from the production boundary. Most countries include estimates for expenditures on software and databases, but the underlying value of the data included in the database is excluded from the production boundary. Since our approach is consistent with national accounting practices, the estimates produced by our method can be incorporated into national accounting frameworks in a straightforward way.

# Labor Costs Estimation and Time-use Proxies

The labor costs approach is a common method for constructing spending estimates when transaction or survey data is limited or not available. For example, many statistical agencies, such as the US Bureau of Economic Analysis, use a labor costs approach to estimate spending levels in a variety of hard-to-measure categories, such as on own-account software production.

Estimating spending levels using a labor costs approach usually requires making some assumptions about the occupations that should be included in the estimate and the amount of time workers in these occupations spend performing relevant tasks. First, the determination of what occupations should be included in the estimate is usually based on whether an occupation performs tasks that are relevant to the spending category. For example, the relevant occupations to consider in a labor costs estimate for own-account software spending includes software developers, database administrators, etc since these occupations are most likely performing tasks related to software development. Second, once these occupations are identified, total hours worked must be scaled by a time-use factor, which reflects how often these occupation engage in these category-relevant tasks. However, time-use factors are rarely observed in practice, and most statistical agencies assume 50 percent in lieu of any estimate derived from data. 

Given occupational time-use data is rarely available, we introduce an alternative approach to labor costs estimation that uses the language contained in job postings to proxy time-use factors. 



# Pre-Processing Steps

## Unpacking the Burning Glass Database

The Burning Glass database consists of more than 200 million job postings from 2010 to 2019. The structured databases are contained in several ``zip`` files, while the unstructured data is contained in XML files. 

# Proxying Time-Use Factors

## Doc2Vec Model Training and Evaluation

To proxy time-use factors, we need to come up with an estimate for the similarity between occupations based on their relationship with data-intensive responsibilities. One way of going about this estimation is by comparing the full set of occupations with known data-intensive occupations using only the language contained in the job posting text. There are a variety of ways to compute document similarities, e.g. TF-IDF, Jaccard Distance, etc., but these bag-of-words methods may miss important semantic similarities between job postings. To this end, we use Doc2Vec to perform our similarity task.

In summary, Doc2vec builds on the Word2Vec algorithm that uses word "contexts" to train a shallow neural network. There are two approaches, continuous bag-of-words and skip-gram. Each approach trains the weights of the neural network to either predict a word's context (skip-gram) or using a word's context to predict itself (continuous bag-of-words). In either case, the trained weights of the neural network correspond to the vector representations of each word in the vocabulary. Doc2Vec builds on this approach by adding a "document tag" that encodes document-level attributes into the Word2Vec model. As a consequence, document vectors are trained alongside word vectors, resulting in a dense vector representation for each document tag.

To dampen the effects of job posting outliers, I train the Doc2Vec model using ONET categories as the document tags. As a consequence, the model trains a vector representation for each ONET category rather than each document. After experimenting with different parameter combinations, I find training the model using the CBOW algorithm, with 1000 dimensions, 15 epochs, a window size of 3 words, and a hierarchical softmax classifier yields the best results. 



Due to limits on our computational resources, I limit the size of the training sample and elect to train a Doc2Vec model for each year in the data. To construct the annual training sample, I start by randomly choosing a week of job postings for a given month. I repeat this sampling process for each month so each month is represented by a weekly job posting file. In other words, our annual training sample is a composite of 12 weekly job postings files. 

After the monthly representatives are sampled and combined into an annual dataset, we split the annual dataset into a training and test sample. As before, our computational resources are limited, and therefore, we randomly sample 1 million job postings from the annual dataset to construct the training sample. We note that for each year we ensure the training and test split is such that the size of the test dataset is at least two-thirds of the training dataset. This is done to ensure an adequate sample remains for validating the trained Doc2Vec model.

After training the Doc2Vec model, I evaluate the model's performance using two common approaches. For each validation procedure, we estimate the performance metric by randomly sampling 50,000 observations with replacement from the test dataset across 5 iterations. Our first metric uses a nearest neighbors criterion to judge the accuracy of the model. For each job posting in the bootstrapped sample, we compute the cosine distance between the inferred job posting's document vector and the representative ONET category embeddings trained by the model. Once computed, we compute a binary metric based on whether the job posting's ONET category is in the 10 nearest ONET category embeddings. We estimate the performance metric by summing up these binary metrics and dividing by the number of job postings in the sample.

Our second approach uses a similarity triplet method. In this approach, we construct a triplet consisting of three job postings. Two of the three job postings are chosen to be in the same ONET category, and the final posting is chosen at random from a different ONET category. To compute the accuracy metric, we assign a binary score to each triplet. The score is equal to one when the cosine distance between the two job postings in the same ONET category is less than the distance between these postings and the job posting from a separate ONET category. The aggregate performance metric is then the sum of all binary scores divided by the total number of triplets constructed from the sample. 

As a complimentary check on the model, I also visualize the trained ONET embeddings using t-SNE (t-distributed Stochastic Neighbor Embeddings). I follow recommended guidelines (such as as PCA to reduce the embedding dimensionality and a large number of iterations) when applying t-SNE to the estimated ONET embeddings for 2011. Visualizing the embeddings in 2 dimensions shows the model appears to be performing reasonably well when preserving local relationships in the higher dimensional space. 


### The Code and Relevant Output

With broad brush strokes laid out above, the details of the program are contained in the code [doc2vec_training_test.py](doc2vec/doc2vec_training_test.py). The sections of the code are pretty self-contained and I have provided detailed to comments that highlight what each line of code is doing. For data on the the size of the training/test datasets for each year, consult the [train_file.csv](Data/train_file.csv) file. A feature file containing vocabulary size and the number of trained embeddings is in [feature_file.csv](Data/feature_file.csv). 

Lastly, the performance file containing the performance metrics is in [performance_file.csv](Data/performance_file.csv). There are four columns of data in the file: ``year, nn_score, nn_rank, triplet_score``. The ``year`` column is self-explanatory. The ``nn_score`` and ``nn_rank`` columns correspond to the nearest neighbor evaluation method discussed above. The ``nn_score`` is the bootstrapped performance metric using the nearest neighbor criteria, and the ``nn_rank`` column gives the average position for instances where the ONET category is in the Top 10 nearest neighbors. Lastly, the ``triplet_score`` column is the bootstrapped performance metric using the triplet method. 

Each trained model is saved on ``serv570`` in the directory ``prediction/burning_glass/Chris/Word2Vec`` (I know the folder name is misleading, but Doc2Vec is basically Word2Vec with a twist!). Unless you have a strong desire to re-train the models, you can use these files to load the models for yourself. 

Finally, the code for constructing the ONET landmark embeddings figure can be found in [tsneLandmarks.py](doc2vec_viz/tsneLandmarks.py), and the figure produced from the code is [doc2vec_landmarks_inset.png](figures/doc2vec_landmarks_inset.png).


## Data-Interfacing Occupations

The second component of the time-use proxy is an estimate for the fraction of jobs in an occupation that interface with data. To identify these jobs, I use Burning Glass' Skills Database. Using the database, we can identify jobs that are likely interfacing with data based on the skills required to perform the job's duties and responsibilities. As an aside, we can also use the same approach to identify jobs that require some software development skillsets. 

Using the Skills database, I perform a keyword search in the ``Skill``, ``SkillsCluster``, and ``SkillClusterFamily`` variables to identify data-related skillsets. The keyword search is performed using the keywords: data, dataset, and database. I manually inspect the results to ensure the search returns relevant skillsets. I execute a similar routine for software development skills. 

With these skills identified, I create an indicator variable that signifies whether a job posting has a data-related skill. With these indicator variables, I use the ONET category for each job posting to construct an estimate for the number of job openings in an ONET category with data-related skills. Dividing this estimate by the total job postings for an ONET category in a year yields the estimate for p_w. 

## Distance to Data-Intensive Job Landmarks

Once the estimate for p_w is constructed, the time-use adjustment factor needs to be proxied. To proxy this factor, I use the ONET embeddings to perform similarity analyses. Performing similarity analyses across all ONET categories would be a silly exercise. For instance, we do not care how similar a truck driver is to a fast food clerk because this tells us nothing about data-related job responsibilities for either occupation. Instead, I want to anchor the similarity analysis around comparisons to "known" data-intensive occupations. In this scenario, we are asking how similar a truck driver is to a data scientist. To select a set of "known" data-intensive occupations, I use p_w as a proxy to sort occupations by their "data-intensiveness". My assumption here is that when a higher fraction of job openings are required to have a data-related skillset, they are more likely to interface with data on routine basis. To reduce random fluctuations, I average p_w over all years and then take the 15 occupations with the highest average p_w. The occupations are as follows:

1. 43-9021.00 - Data Entry Keyers
2. 15-1141.00 - Database Administrators
3. 15-1199.06 - Database Architects
4. 15-1199.07 - Data Warehousing Specialists
5. 19-4061.00 - Social Science Research Assistants
6. 15-1111.00 - Computer and Information Research Scientists
7. 15-2041.01 - Biostatisticians
8. 15-2041.00 - Statisticians
9. 15-1199.08 - Business Intelligence Analysts
10. 19-1029.01 - Bioinformatics Scientists
11. 43-9111.00 - Statistical Assistants
12. 19-3022.00 - Survey Researchers
13. 29-2092.00 - Hearing Aid Specialists
14. 15-1199.05 - Geographic Information Systems Technicians
15. 15-2041.02 - Clinical Data Managers

The majority of these make sense, with the exception of Hearing Aid Specialists, which may warrant going back to the data-related skillsets and fine-tune the selections a bit. Nevertheless, we can compute the distance between each ONET embedding category and these data-intensive landmark embeddings, and this distance serves as proxy for the time-use adjustment factor. To compute the distance, we compute the cosine distance between each ONET embedding and the data-intensive landmark embeddings. We then use the minimum cosine distance as a proxy for the time-use adjustment factor. This exercise is the equivalent of finding the nearest data-intensive landmark to each ONET embedding. 

### A Brief Aside

Before moving on, why do I think this works? The simple algebraic equivalence shows the time-use factor can be decomposed into two components. The first component the time-use adjustment factor measures the relative, average frequency an occupation is working with data. The second component, p_w, is an estimate for the percentage of workers in an occupation that are working with data in some capacity. Since the decomposition is multiplicative, these two components work to re-inforce or dampen the effects of each individual component. For example, even though an occupation may reside nearby a landmark data occupation, this does not necessarily imply a higher estimate for time-use since very few workers may be interfacing with data, and vice versa. 


Here is a list of the top 20 occupations (that are not data-intensive occupations) based on time-use:

1. 19-3099.00 - Social Scientists and Related Workers, All Other -  0.27 
2. 15-1199.04 - Geospatial Information Scientists and Technologists - 0.20
3. 15-1132.00 - Software Developers, Applications - 0.19
4. 15.1133.00 - Software Developers, Systems Software - 0.18
5. 15.1121.00 - Computer Systems Analysts - 0.18
6. 17-3031.02 - Surveying and Mapping Technicians- 0.18
7. 15-1131.00 - Computer Programmers - 0.17
8. 13-1111.00 - Management Analysts  - 0.15
9. 19-1041.00 - Epidemiologists - 0.15
10. 17-1021.00 - Cartographers and Photogrammetrists - 0.13
11. 15-2031.00 - Operations Research Analysts - 0.13
12. 11-9121.01 - Clinical Research Coordinators - 0.13
13. 13-2099.01 - Financial Quantitative Analysts - 0.13
14. 15-1199.02 - Computer Systems Engineers/Architects - 0.12
15. 19-3011.00 - Economists - 0.12 
16. 19-1020.01 - Biologists 0.11
17. 11-9121.00 - Natural Sciences Managers - 0.11 
18. 19-1042.00 - Medical Scientists, Except Epidemiologists - 0.10
19. 15-1199.00 - Computer Occupations, All Other - 0.10
20. 15-1134.00 - Web Developers - 0.09

Based on these estimates, I garner that we are being very conservative in how we estimate time-use factors, especially given the base assumption is 50 percent. 

# Labor Costs Estimate

With estimates for the time-use adjustment factor and p_w, we are ready to construct the labor costs estimate. At this point, the labor costs estimate is straightforward. We use OES data to collect information and salary and employment for each SOC category (note: even though the analysis is conducted for ONET codes, we must aggregate this to SOC codes for consistency with the OES data. 
