import torch
from transformers import BertModel

class NQModel(torch.nn.Module): 
    def __init__(self):
        super(NQModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.AT_1 = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        self.relu = torch.nn.ReLU()
        self.AT_2 = torch.nn.Linear(in_features=512, out_features=3, bias=True)
        
        self.end_v   = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        self.start_v = torch.nn.Linear(in_features=512*768, out_features=512, bias=True)
        
        self.YN = torch.nn.Linear(in_features=768, out_features=3, bias=True)
        
    def forward(self, inp_ids, attn_mask, token_types):
        op_all, op_first = self.bert(inp_ids, attn_mask, token_types)
        
        op_flat = op_all.view(op_all.shape[0], -1)
        
        ans_type = self.AT_2(self.relu(self.AT_1(op_flat)))
        start = self.start_v(op_flat)
        end  = self.end_v(op_flat)
        yes_no = self.YN(op_first)
        
        return (ans_type, start, end, yes_no)