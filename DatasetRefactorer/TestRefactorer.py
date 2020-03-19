import DatasetRefactorer.Constants as Constants
from DatasetRefactorer.Example import ExampleCreator
from tqdm import tqdm
import json

def refactor_test() : 
	output_file = open(Constants.test_path,"w")

	with open(Constants.raw_test_path,"r") as test_file : 
	    for line in tqdm(test_file) : 
	        data = json.loads(line)
	        test_examples = ExampleCreator.test_item_to_examples(data)
	        test_examples = [json.dumps(example.to_dict()) for example in test_examples]
	        for item in test_examples : 
	            output_file.write(item + '\n')

	output_file.close()