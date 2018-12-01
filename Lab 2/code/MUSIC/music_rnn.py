import torch
import torch.nn as nn
from data_loader_music import *
from torch.utils.data import DataLoader


class LSTM(nn.Module):
	"""

	"""
	def __init__(self, input_size = 4, hidden_size = 20):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, batch_first=True)
		self.fc = nn.Linear(self.hidden_size, 4)

	def forward(self, input):
		"""
		
		"""
		encoder_outputs, _ = self.rnn(input, torch.randn(1, input.size(0), self.hidden_size) ) 
		encoder_outputs = encoder_outputs[:, -1:, :]
		outputs = self.fc(encoder_outputs.squeeze())
		
		return outputs

#dataloading part
train_data = Train()
train_loader = DataLoader(train_data, batch_size = 64, shuffle = False, num_workers = 2)
train_data_len = len(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM()

model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

num_epochs = 500
best_loss = float("inf")

for epoch in range(num_epochs):

	epoch_loss = 0.0
	model.train()
	for i, (ip,op) in enumerate(train_loader):
		ip = ip.to(device)
		op = op.to(device)

		outputs = model(ip) 

		loss = criterion(outputs, op)

		#backprop here
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss += (loss.data * ip.shape[0]) 
		
	epoch_loss = (epoch_loss / float(train_data_len))
	
	print("Ep Train: {}/{}, ls: {}".format(epoch, num_epochs, epoch_loss.data))  

	if (epoch_loss <= best_loss ):
		best_loss = epoch_loss
		state = {'state_dict':model.state_dict()}
		torch.save(state, "./music_state_rnn.pt" )
	
print (best_loss)
best_state = torch.load("music_state_rnn.pt")
model.load_state_dict(best_state['state_dict'])

#generating the file
train_loader = DataLoader(train_data, batch_size = 1, shuffle = False, num_workers = 2)
model.eval()
for i, (ip,op) in enumerate(train_loader):
	ip = ip.to(device)
	op = op.to(device)

pred_outputs = None

for j in range(100):
	outputs = model(ip) 
	ip = torch.cat((ip[:,1:,:], outputs.unsqueeze(0).unsqueeze(0)), dim = 1)
	if (j == 0):
		pred_outputs = outputs.unsqueeze(0)
	else:
		pred_outputs = torch.cat((pred_outputs, outputs.unsqueeze(0)), dim = 0)


mean, std = torch.zeros(1), torch.zeros(1)
'''mean[0], mean[1], mean[2], mean[3] = 59.93, 50.91, 39.42, 36.19
std[0], std[1], std[2], std[3] = 26.91, 19.67, 21.11, 16.41'''

mean[0] = 44.367
std[0] = 22.38

pred_outputs *= std
pred_outputs += mean


pred_outputs = pred_outputs.data.numpy()

f = open("final_output_rnn.txt", 'w')

for point in (pred_outputs):
	f.write(str(int(point[0])) + " " + str(int(point[1])) + " " + str(int(point[2])) + " " + str(int(point[3])) + "\n")
f.close()
