from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine as cosine_distance


# Function for computing the minimum distance to a landmark vector
def min_dist(x,vecs=dataONET_vecs):

	# Start by inferring the document embedding
	x_vec = model.infer_vector(simple_preprocess(x))


	# Return the minimum distance to a landmark vectors
	return np.min([cosine_distance(x_vec,data_vec) for data_vec in dataONET_vecs])

