import torch
from transformers import BertModel

class NQModel(torch.nn.Module): 
    def __init__(self):
        super(NQModel, self).__init__()
        self.AT_1 = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        self.relu = torch.nn.ReLU()
        self.AT_2 = torch.nn.Linear(in_features=512, out_features=3, bias=True)
        
        self.end_v   = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        self.start_v = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        
        self.YN_1 = torch.nn.Linear(in_features=512*768, out_features=3, bias=True)
        self.YN_2 = torch.nn.Linear(in_features=512, out_features=3, bias=True)
        
    def forward(self, bert_encoding):
        encoding_flat = op_all.view(bert_encoding.shape[0], -1)
        
        ans_type = self.AT_2(self.relu(self.AT_1(encoding_flat)))
        start = self.start_v(encoding_flat)
        end  = self.end_v(encoding_flat)
        yes_no = self.YN_2(self.YN_1(encoding_flat))
        
        return (ans_type, start, end, yes_no)