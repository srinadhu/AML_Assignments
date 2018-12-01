"""
DataLoader for timeseries dataset
"""

import numpy as np 

import torch
from torch.utils.data.dataset import Dataset


class Train(Dataset):
	"""
	dataset class for train part.
	"""
	def __init__(self):

		self.data_pts_dct = {}
		f = open("pre_train.txt", 'r')
		for line in f:
			index, value = line.rstrip().split(",")
			self.data_pts_dct[int(index)] = float(value)

		self.length = 0
		self.train_pts = []
		for i in range(1, 5001):
			if (i in self.data_pts_dct.keys() and (i+20) in self.data_pts_dct.keys()): #for both input and output
				self.length += 1	
				self.train_pts.append(i)

	def __getitem__(self, index):
		start = self.train_pts[index]
		
		input = np.zeros([20, 1])		
		for i in range(start, start + 20):
			input[i-start] = self.data_pts_dct[i]

		output = np.zeros(1)
		output[0] = self.data_pts_dct[start + 20]

		input, output = torch.from_numpy(input).float(), torch.from_numpy(output).float()

		return (input, output)

	def __len__(self): 
		return self.length

class Val(Dataset):
	"""
	dataset class for train part.
	"""
	def __init__(self):

		self.data_pts_dct = {}
		f = open("pre_train.txt", 'r')
		for line in f:
			index, value = line.rstrip().split(",")
			self.data_pts_dct[int(index)] = float(value)
		f.close()

		self.val_pts_dct = {}
		f = open("pre_val.txt", 'r')
		for line in f:
			index, value = line.rstrip().split(",")
			self.val_pts_dct[int(index)] = float(value)
		f.close()

		self.train_pts = []
		for i in range(1, 5001):
			if (i in self.val_pts_dct.keys() and (i+19) in self.val_pts_dct.keys()): #for both input and output	
				self.train_pts.append(i)

	def __getitem__(self, index):
		start = self.train_pts[index]
		
		input = np.zeros([20, 1])		
		for i in range(start - 20, start):
			input[i-start + 20] = self.data_pts_dct[i]

		output = np.zeros(20)		
		for i in range(start, start + 20):
			output[i-start] = self.val_pts_dct[i]

		input, output = torch.from_numpy(input).float(), torch.from_numpy(output).float()

		return (input, output)

	def __len__(self): 
		return 5

class Test(Dataset):
	"""
	dataset class for test part.
	"""
	def __init__(self):

		self.data_pts_dct = {}
		f = open("pre_val.txt", 'r')
		for line in f:
			index, value = line.rstrip().split(",")
			self.data_pts_dct[int(index)] = float(value)
		f.close()

		self.val_pts_dct = {}
		f = open("pre_test.txt", 'r')
		for line in f:
			index, value = line.rstrip().split(",")
			self.val_pts_dct[int(index)] = float(value)
		f.close()

		self.train_pts = []
		for i in range(1, 5001):
			if (i in self.val_pts_dct.keys() and (i+19) in self.val_pts_dct.keys()): #for both input and output	
				self.train_pts.append(i)

	def __getitem__(self, index):
		start = self.train_pts[index]
		
		input = np.zeros([20, 1])		
		for i in range(start - 20, start):
			input[i-start + 20] = self.data_pts_dct[i]

		output = np.zeros(20)		
		for i in range(start, start + 20):
			output[i-start] = self.val_pts_dct[i]

		input, output = torch.from_numpy(input).float(), torch.from_numpy(output).float()

		return (input, output)

	def __len__(self): 
		return 5