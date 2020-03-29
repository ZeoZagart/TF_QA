import torch

class LossFn: 
	def __init__(self, device):
	    self.ans_type = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.4, 0.5]).to(device))
    	self.start = torch.nn.BCEWithLogitsLoss()
    	self.end = torch.nn.BCEWithLogitsLoss()
    	self.yes_no = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.45, 0.45]).to(device))