import json
from Constants import *

class TrainExample : 
	long_ans: str        = None
	short_ans: str       = None
	yes_no_ans: bool     = False
	question_text: str   = None
	is_ans_correct: bool = False

	def __init__(self, long_ans: str, short_ans: str,
					yes_no_ans: bool, question_text: str, is_correct: bool) : 
		self.long_ans       = long_ans
		self.short_ans      = short_ans
		self.yes_no_ans     = yes_no_ans
		self.question_text  = question_text
		self.is_ans_correct = is_correct      

	def __repr__(self) : 
		return json.dumps(self.to_dict())

	def to_list(self) : 
		return [self.question_text, self.long_ans, self.short_ans, self.yes_no_ans, self.is_ans_correct]

	def to_dict(self) : 
		return {"question_text": self.question_text, "long_ans": self.long_ans, "short_ans": self.short_ans, "yes_no": self.yes_no_ans, "is_correct": self.is_ans_correct}

class TestExample : 
	question_id: str   = None
	question_text: str = None
	long_ans: str      = None

	def __init__(self, question_id: str, question_text: str, long_ans: str) : 
		self.question_id = question_id
		self.question_text = question_text
		self.long_ans = long_ans

	def to_dict(self) : 
		return {"question_id": self.question_id, "question_text": self.question_text, "long_ans": self.long_ans}

class ExampleCreator : 
	def test_item_to_examples(data_item) : 
		test_set    = []
		question_id = data_item[EXAMPLE_ID]
		question    = data_item[QUESTION_TEXT]
		document    = data_item[DOCUMENT_TEXT].split()

		for idx, candidate in enumerate(data_item[LONG_ANSWER_CANDIDATES]) : 
			if candidate[TOP_LEVEL] == False : continue
			long_ans = ExampleCreator.get_string_from_token_list(
				document[candidate[START_TOKEN]:candidate[END_TOKEN]])
			test_set.append(TestExample(question_id, question, long_ans))
		return test_set


	def train_item_to_examples(data_item) : 
		train_set = []
		question = data_item[QUESTION_TEXT]
		document = data_item[DOCUMENT_TEXT].split()

		long_ans_idx = ExampleCreator.get_outermost_long_ans_index(data_item[ANNOTATIONS][0],data_item[LONG_ANSWER_CANDIDATES])

		for idx, candidate in enumerate(data_item[LONG_ANSWER_CANDIDATES]) : 
			if candidate[TOP_LEVEL] == False : continue
			if idx == long_ans_idx : 
				[long_ans, short_ans, yes_no_ans] = ExampleCreator.get_ans_from_annotation(data_item[ANNOTATIONS][0],document)
				train_set.append(TrainExample(long_ans, short_ans, yes_no_ans, question, True))
			else :
				long_ans = ExampleCreator.get_string_from_token_list(
					document[candidate[START_TOKEN]:candidate[END_TOKEN]])
				train_set.append(TrainExample(long_ans, None, None, question, False))
		return train_set

	def get_outermost_long_ans_index(annotations, long_ans_candidates) : 
		long_ans = annotations[LONG_ANSWER]
		if long_ans == None or len(long_ans) == 0 : return -1

		candidate_idx = long_ans[CANDIDATE_IDX]
		while long_ans_candidates[candidate_idx][TOP_LEVEL] == False : candidate_idx -= 1

		return candidate_idx

	def get_ans_from_annotation(annotations, document) : 
		long_string  = None
		short_string = None

		if len(annotations[LONG_ANSWER]) > 0 : 
			long_start = annotations[LONG_ANSWER][START_TOKEN]
			long_end   = annotations[LONG_ANSWER][END_TOKEN]
			long_string = ExampleCreator.get_string_from_token_list(document[long_start: long_end])

		if len(annotations[SHORT_ANSWER]) > 0 : 
			short_start = annotations[SHORT_ANSWER][0][START_TOKEN]
			short_end   = annotations[SHORT_ANSWER][0][END_TOKEN]
			short_string = ExampleCreator.get_string_from_token_list(document[short_start: short_end])

		yes_no_ans = annotations[YES_NO]

		return [long_string, short_string, yes_no_ans]

	def get_string_from_token_list(tokens) :
		token_join = ' '.join([token for token in tokens if token[0] != '<'])
		return '<start> ' + token_join + ' <end>'
