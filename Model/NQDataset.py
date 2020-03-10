import torch
import transformers
from transformers import BertTokenizer
from typing import List
from torch.utils.data import Dataset
from Model.ModelConstants import *

def error(function, message) : 
	print("error in :", function)
	print("message :", message)


class NQDataset(Dataset) : 
	def __init__(self, dataset) : 
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.data = dataset

	def __len__(self) : 
		return len(self.data)

	def __getitem__(self, index) : 
		item = self.data[index]
		tokenized = self.tokenizer.encode_plus(item.question_text, item.long_ans, **tokenizer_config)

		# the inputs to bert
		inputids, token_type, mask = tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask'] 

		# the outputs expected
		answer_type = self.get_ans_type(item)
		short_start, short_end = self.get_short_start_end(item.short_ans, inputids.tolist()[0])
		yes_no = torch.LongTensor([item.yes_no_ans or 0])

		return inputids, token_type, mask, answer_type, short_start, short_end, yes_no

	def get_ans_type(self, item) : 
		'''
		answer type : 
			1 -> it is wrong answer 
			2 -> it is short answer
			3 -> it is yes no answer,
		each one is exclusive, so we return a one-hot encoded list of size 3
		'''
		ans_type = [0]*3
		if item.is_ans_correct == False : ans_type[0]  = 1
		elif item.yes_no_ans is not None : ans_type[2] = 1
		elif item.short_ans is not None : ans_type[1]  = 1
		else : error("get_ans_type in NQDataset", "no answer type found")
		return torch.LongTensor([ans_type])

	def get_short_start_end(self, short_ans_list: List[str], long_list: List[int]) : 
		'''
		calculates the span of each short ans string in the form of one-hot-encoded start and end positions 
		inside the tokenized long ans string
		'''
		if short_ans_list == None : short_ans_list = []
		short_ans_list = [self.get_span(short_str, long_list) for short_str in short_ans_list]
		short_ans_list = [item for item in short_ans_list if item is not None]

		start_list, end_list = [0]*sentence_len, [0]*sentence_len

		for start, end in short_ans_list : 
			start_list[start] = 1
			end_list[end] = 1

		return torch.LongTensor([start_list]), torch.LongTensor([end_list])


	def get_span(self, short_string: str, long_list: List[int]) : 
		short_list = self.tokenizer.encode(short_string)

		for idx, value in enumerate(long_list) : 
			if short_list[0] == long_list[idx] and \
				idx + len(short_list) <= len(long_list) and \
				short_list[-1] == long_list[idx + len(short_list) - 1] : 
					return (idx, idx + len(short_list) - 1)
		return None