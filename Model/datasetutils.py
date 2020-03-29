import json
import torch
from tqdm import tqdm_notebook as tqdm
from Model import ModelConstants
from Model.NQDataset import NQDataset
from torch.utils.data import DataLoader
from DatasetRefactorer import Constants
from DatasetRefactorer.Example import TrainExample

def get_dataset(device = torch.device("cpu"), num_workers = 0) : 
	with open(Constants.train_path,"r") as train_file : 
		dataset = [TrainExample(**(json.loads(line))) for line in tqdm(train_file)]


	nqdataset = NQDataset(dataset)
	train_len = int(len(nqdataset)*0.95)
	valid_len = len(nqdataset) - train_len
	trainData, valData = torch.utils.data.random_split(nqdataset, [train_len, valid_len] )
	traingen = DataLoader(trainData, num_workers = num_workers, **ModelConstants.dataset_params)
	validgen = DataLoader(valData, num_workers = num_workers, **ModelConstants.dataset_params)

	return (traingen, validgen)