from DatasetRefactorer import Constants
from DatasetRefactorer.Example import ExampleCreator
from tqdm import tqdm
import json
import multiprocessing 
import gc

def worker(data, writer, lock) : 
	result_set = []
	for item in data : 
		result_set += ExampleCreator.train_item_to_examples(item)
	lock.acquire()
	for item in result_set : 
		writer.write(json.dumps(item.to_dict()) + '\n')
	lock.release()

def refactor_train() : 
	dataset = []
	queue   = []
	op_file = open(Constants.train_path,"w")

	lock = multiprocessing.Lock()
	with open(Constants.raw_train_path, 'r') as train_file : 
		for i, line in tqdm(enumerate(train_file)) : 
			dataset.append(json.loads(line))
			if i%5000 == 0 : 
				work = multiprocessing.Process(target=worker, args=(dataset, op_file, lock))
				work.start()
				queue.append(work)
				dataset = []

	for work in tqdm(queue) : 
		work.join()

	op_file.close()