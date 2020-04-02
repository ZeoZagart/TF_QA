import torch

class LossFn: 
	def __init__(self, device):
		self.ans_type = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 90000, 100005]).to(device))
		self.start = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100005]*512, dtype = torch.float).to(device))
		self.end = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100005]*512, dtype = torch.float).to(device))
		self.yes_no = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 100005, 100005]).to(device))