import torch

class LossFn: 
    loss_AT = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.4, 0.5]))
    loss_start = torch.nn.BCEWithLogitsLoss()
    loss_end = torch.nn.BCEWithLogitsLoss()
    loss_yes_no = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.45, 0.45]))