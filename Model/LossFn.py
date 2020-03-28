import torch

class LossFn: 
	def __init__(self, device)
	    ans_type = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.4, 0.5]).to(device))
    	start = torch.nn.BCEWithLogitsLoss()
    	end = torch.nn.BCEWithLogitsLoss()
    	yes_no = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.45, 0.45]).to(device))