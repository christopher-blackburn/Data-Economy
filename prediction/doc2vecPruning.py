import pandas as pd
import numpy as np 
from doc2vec_model import import_doc2vec
from annual_sample import annual_sample
import landmarkDist 
from tqdm.contrib.concurrent import process_map
from dataDictionary import dataTitle, dataText, dataJob
from functools import reduce




if __name__ == '__main__':

	'''
	------------------------------------------------------------
	Create the landmark data vectors
	------------------------------------------------------------
	In this step, I need to read the annual files to select
	the appropriate landmarks. For now, I am choosing
	the top 15 occupations.

	Furthermore, I'm going to drop any occupations
	with less than 5 job postings in the BG dataset.
	'''

	print('Creating the landmark vectors',end='...')
	# Specify the path to the dataset
	annua_data_path = 'E:/Research1/prediction/Burning_glass/Chris/Output/dictEstimation{}.csv'.format(2011)

	# Load the annual estimates created from the dictionary-based methods
	annual_estimates = pd.read_csv(annual_data_path)

	# Extract the top 15 ONET codes
	dataONET = list(annual_estimates.sort_values(by='onet_datatitle_share',ascending=False)['ONET'].head(15))

	# Import the trained Doc2Vec model
	model = import_doc2vec(2011)

	# Extract the model's tags (these are the ONET category landmarks)
	model_tags = model.docvecs.index2entity

	# Extract the landmark vectors for the data categories we have identified
	dataONET_vecs = [model.docvecs[onet_code] for onet_code in dataONET]

	print('Landmark vectors successfully constructed!')

	'''
	------------------------------------------------------------
	Compute minimum landmark distance
	------------------------------------------------------------
	In this step, I compute each document's minimum distance
	to a landmark.
	'''

	print('Constructing the annual dataset',end='...')

	# Load the annual BG data
	annualDF = annual_sample(2011)

	print('Annual data successfully constructed.')

	# Drop any missing values in the title or text
	annualDF = annualDF[annualDF['CleanTitle'].notna()]
	annualDF = annualDF[annualDF['jobText'].notna()]


	print('Computing minimum distance to landmark vectors',end='...')
	
	# Compute the minimum landmark distance
	annualDF['min_dist'] = process_map(landmarkDist.min_dist,list(annualDF['jobText']))

	print('Minimum distance estimated.')


	'''
	------------------------------------------------------------
	Identify data in title and text
	------------------------------------------------------------
	In this step, I identify jobs with data in the title and 
	text
	'''

	print('Performing the standard dictionary-based estimation',end='...')
	# Data related job titles
	annualDF['dataTitle'] = annualDF['CleanTitle'].apply(lambda x: dataTitle(x))

	# Data within the job text 
	annualDF['dataText'] = process_map(dataText,list(annualDF['jobText']))

	print('Dictionary-based estimation completed.')


	'''
	------------------------------------------------------------
	Perform the pruning
	------------------------------------------------------------
	In this step, I compute each document's minimum distance
	to a landmark. Before I begin, I will also save some
	data on the histogram.
	'''
	# Histogram save path
	histSave = 'E:/Research1/prediction/Burning_glass/Chris/Output/landmarkHist{}.csv'.format(2011)

	# Pruned data save path 
	pruneSave = 'E:/Research1/prediction/Burning_glass/Chris/Output/doc2vecPruning{}.csv'.format(2011)

	# Extract the histogram information
	H, bins = np.histogram(annualDF['min_dist'],bins=50)

	# Save the histogram information
	pd.DataFrame({'height':H,'bins':bins[:-1]}).to_csv(histSave,index=False)

	# Create a list of thresholds 
	thresholds = [0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95]

	print('Pruning data using preset threshold values',end='...')

	# Loop through the thresholds and compute jobs that come in under the threshold values
	threshold_data = [annualDF[annualDF['min_dist'] <= thresh].groupby(['ONET'])['dataText'].sum().reset_index() for tresh in thresholds]

	# Merge the threshold datasets together
	thresh_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ONET'],how='outer'), treshold_data)

	# Save the pruned data
	thresh_merged.to_csv(pruneSave,index=False)

	print('Pruned data saved successfully.')




