import torch
import torch.nn as nn
from data_loader import *
from torch.utils.data import DataLoader


class LSTM(nn.Module):
	"""

	"""
	def __init__(self, input_size = 1, hidden_size = 20):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first=True)
		self.fc = nn.Linear(self.hidden_size, 1)

	def forward(self, input):
		"""
		
		"""
		encoder_outputs, _ = self.rnn(input, (torch.randn(1, input.size(0), self.hidden_size),torch.randn(1, input.size(0), self.hidden_size) )  )
		encoder_outputs = encoder_outputs[:, -1:, :]
		outputs = self.fc(encoder_outputs.squeeze())
		
		return outputs

#dataloading part
train_data = Train()
train_loader = DataLoader(train_data, batch_size = 64, num_workers = 2)
train_data_len = len(train_data)

val_data = Val()
val_loader = DataLoader(val_data, batch_size = 5)

test_data = Test()
test_loader = DataLoader(test_data, batch_size = 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM()

model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

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

	model.eval()
	for i, (ip,op) in enumerate(val_loader):
		ip = ip.to(device)
		op = op.to(device)

		pred_outputs = None

		for j in range(20):
			outputs = model(ip) 
			ip = torch.cat((ip[:,1:,:], outputs.unsqueeze(1)), dim = 1)
			if (j == 0):
				pred_outputs = outputs
			else:
				pred_outputs = torch.cat((pred_outputs, outputs), dim = 1)

		loss = criterion(pred_outputs, op)
	
	print("Ep val: {}/{}, ls: {}".format(epoch, num_epochs, loss.data)) 

	if (loss <= best_loss ):
		best_loss = loss
		state = {'state_dict':model.state_dict()}
		torch.save(state, "./state_lstm.pt" )
	
print (best_loss)
best_state = torch.load("state_lstm.pt")
model.load_state_dict(best_state['state_dict'])

model.eval()
for i, (ip,op) in enumerate(test_loader):
	ip = ip.to(device)
	op = op.to(device)

	pred_outputs = None

	for j in range(20):
		outputs = model(ip) 
		ip = torch.cat((ip[:,1:,:], outputs.unsqueeze(1)), dim = 1)
		if (j == 0):
			pred_outputs = outputs
		else:
			pred_outputs = torch.cat((pred_outputs, outputs), dim = 1)

	loss = criterion(pred_outputs, op)
	
print("Final Testing, ls: {}".format(loss.data)) 