"""
Module that filters the dataset and the corresponding image and 
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")


import h5py
import argparse
from tqdm import tqdm
import json

def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help="prefix of the QA dataset json file")
	parser.add_argument('--data', type=str, help="path to the data directory where the QA json file and image features are stored")

	return parser.parse_args()

def filter(args, split):

	"""
	Implement the filtering criteria to filter the QA pairs here and return the list of images satisfying that criteria.
	"""
	
	# first filter the train file
	data_path = os.path.join(args.data, "{}_{}.json".format(args.dataset, split))

	with open(data_path, 'r') as f:
		data = json.load(f)
		
	# Current Function is to first filter all the questions of query type

	filtered_data = {'questions': []}

	filtered_data['questions'] = [x for x in data['questions'] and x['type'] == 'query']

	print(len(filtered_data['questions']))
	
	images = [ x['imageId'] for x in filtered_data['questions'] ] 
	
	print(len(images))
	return images

if __name__ == "__main__":

	args = get_args()