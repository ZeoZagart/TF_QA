import json
from Constants import *

class Example : 
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
		return ("Example : " + 
			"\n longAns      : " + self.long_ans           + 
			"\n shortAns     : " + str(self.short_ans)     + 
			"\n yesNoAns     : " + str(self.yes_no_ans)    + 
			"\n questionText : " + str(self.question_text) +
			"\n isCorrect    : " + str(self.is_ans_correct))

class ExampleCreator : 
	def train_item_to_examples(data_item) : 
		example_set = []
		long_ans, short_ans, yes_no_ans = None, None, None
		question = data_item[QUESTION_TEXT]
		document = data_item[DOCUMENT_TEXT].split()

		long_ans_idx = ExampleCreator.get_outermost_long_ans_index(data_item[ANNOTATIONS][0],data_item[LONG_ANSWER_CANDIDATES])

		for idx, candidate in enumerate(data_item[LONG_ANSWER_CANDIDATES]) : 
			if candidate[TOP_LEVEL] == False : continue
			if idx == long_ans_idx : 
				[long_ans, short_ans, yes_no_ans] = ExampleCreator.get_ans_from_annotation(data_item[ANNOTATIONS][0],document)
				example_set.append(Example(long_ans, short_ans, yes_no_ans, question, True))
			else :
				long_ans = ExampleCreator.get_string_from_token_list(
					document[candidate[START_TOKEN]:candidate[END_TOKEN]])
				example_set.append(Example(long_ans, None, None, question, False))
		return example_set

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
