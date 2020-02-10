import Constants
from Example import ExampleCreator
from tqdm import tqdm
import json

test_file_path = Constants.test_file_path
test_op_path   = Constants.test_op_path

output_file = open(test_op_path,"w")

with open(test_file_path,"r") as test_file : 
    for line in tqdm(test_file) : 
        data = json.loads(line)
        test_examples = ExampleCreator.test_item_to_examples(data)
        test_examples = [json.dumps(example.to_dict()) for example in test_examples]
        for item in test_examples : 
            output_file.write(item + '\n')

output_file.close()