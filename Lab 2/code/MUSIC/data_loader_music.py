"""
DataLoader for music timeseries dataset
"""

import numpy as np 

import torch
from torch.utils.data.dataset import Dataset

class Train(Dataset):
	"""
	dataset class for train part.
	"""
	def __init__(self):

		with open('./pre_F.txt', 'r') as f:
			self.lines = [l.strip().split() for l in f.readlines()]
			self.data = [[float(i) for i in l] for l in self.lines]
			self.input = [self.data[i:i+20] for i in range(len(self.data)-20)]
			self.output = [self.data[i+20] for i in range(len(self.data)-20)]
			self.input =  np.array(self.input)
			self.output =  np.array(self.output)
			self.length = len(self.input)

	def __getitem__(self, index):
		
		input = self.input[index]
		output = self.output[index]

		input, output = torch.from_numpy(input).float(), torch.from_numpy(output).float()

		return (input, output)

	def __len__(self): 
		return self.length