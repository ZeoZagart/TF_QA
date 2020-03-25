from Model.NQModel import NQModel
from Model.LossFn import LossFn
import torch
import time
import sklearn
import datetime
import Model.datasetutils as datasetutils
import Model.tensorboardutils as boardutils
import torch.utils.tensorboard as tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm_notebook as tqdm
import transformers

TensorBoardLocation = 'runs/NQ_TIME:{}'.format(int((time.time() - 1583988084)/60))
print (" ~~~~~~~~~~ Board Location : " + TensorBoardLocation)

epochs = 1 # no loop 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = tensorboard.SummaryWriter(TensorBoardLocation)

traingen, validgen = datasetutils.get_dataset(device)

print (" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataset Fetched ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")

num_steps = len(traingen)
val_steps = len(validgen)

model = NQModel()

print (" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model Fetched ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")

dist_init = dist.init_process_group(dist.Backend.GLOO,init_method='file:shared_file' ,world_size=6, rank = 0, timeout=datetime.timedelta(0,15))

print (" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Process Group Initiated ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")

model_parallel = DistributedDataParallel(model)

print (" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Parallel Model Created ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
optim = transformers.AdamW(model_p.parameters())
scheduler = transformers.get_cosine_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=800,num_cycles=0.5, last_epoch=-1)

AnswerTypes = ['Wrong Ans', 'Short Ans', 'Yes No']
YesNoLabels = ['No', 'Yes']

def update_confusion_matrix(ATMatrix, YNMatrix, StartM, EndM, output, target) : 
	predsT = output[0].argmax(dim = 1)
	truthT = target[0].argmax(dim = 1)

	for x, y in zip(predsT, truthT) : 
		ATMatrix[x][y] += 1


	predsYN = (torch.sigmoid(output[3].flatten()) >= 0.5) + 1 -1
	truthYN = target[3].flatten()

	for x, y in zip(predsYN, truthYN) : 
		YNMatrix[x][y] += 1    


	start01 = (torch.sigmoid(output[1].flatten()) >= 0.5) +1 -1
	end01   = (torch.sigmoid(output[2].flatten()) >= 0.5) +1 -1

	startcm = sklearn.metrics.confusion_matrix(target[1].flatten().detach().numpy(), start01)
	endcm   = sklearn.metrics.confusion_matrix(target[2].flatten().detach().numpy(), end01)

	StartM += torch.from_numpy(startcm)
	EndM   += torch.from_numpy(endcm)    


def log_confusion_matrix(matrix, labels, name, step): 
	opfigure = boardutils.confusion_matrix_image(matrix.detach().numpy(), labels)
	writer.add_figure(name, opfigure, step)
	
def log_matrices(AnsTypeM, YNM, StM, EndM, call_type, steps):
	log_confusion_matrix(AnsTypeM, AnswerTypes, "Answer type confusion matrix" + call_type, steps)
	log_confusion_matrix(YNM, YesNoLabels, "Yes No confusion matrix" + call_type, steps)
	log_confusion_matrix(StM, YesNoLabels, "Start confusion matrix" + call_type, steps)
	log_confusion_matrix(EndM, YesNoLabels, "End confusion matrix" + call_type, steps) 


def validate(val_num) : 
	model.eval()
	
	ValAnswerTypeMatrix = torch.zeros([3, 3], requires_grad = False)
	ValYesNoMatrix      = torch.zeros([2, 2], requires_grad = False)
	ValStartMatrix      = torch.zeros([2, 2], requires_grad = False)
	ValEndMatrix        = torch.zeros([2, 2], requires_grad = False)
	
	at_l, start_l, end_l, yn_l = 0,0,0,0
	
	with torch.no_grad():
		cur_val_step
		for inp_id, inp_type, inp_mask, ans_type, start, end, yes_no in tqdm(validgen) : 
			output = model(inp_id.squeeze(), inp_mask.squeeze(), inp_type.squeeze())  

			update_confusion_matrix(ValAnswerTypeMatrix, ValYesNoMatrix, ValStartMatrix, ValEndMatrix, output, (ans_type, start, end, yes_no))

			## Calculate Loss
			at_l += LossFn.loss_AT(output[0], ans_type.squeeze().argmax(1)).item()
			start_l += LossFn.loss_start(output[1], start.squeeze().type(torch.FloatTensor)).item()
			end_l += LossFn.loss_end(output[2], end.squeeze().type(torch.FloatTensor)).item()
			yn_l += LossFn.loss_yes_no(output[3], yes_no.squeeze()).item()
			
			
	## Save loss values
	writer.add_scalars('Loss values Validation',
		{"AT_loss_val" : at_l,"Start_loss_val":start_l, "End_loss_val":end_l, "Yes_no_loss_val":yn_l},
		val_num, time.time())

	log_matrices(ValAnswerTypeMatrix, ValYesNoMatrix, ValStartMatrix, ValEndMatrix, " eval", val_num)    


def train() : 
	AnswerTypeMatrix = torch.zeros([3,3], requires_grad = False)
	YesNoMatrix      = torch.zeros([2,2], requires_grad = False)
	StartMatrix      = torch.zeros([2,2], requires_grad = False)
	EndMatrix        = torch.zeros([2,2], requires_grad = False)

	start = time.time()
	model.train()
	steps = -1

	for inp_id, inp_type, inp_mask, ans_type, start, end, yes_no in tqdm(traingen) : 
		steps += 1
		output = model(inp_id.squeeze(), inp_mask.squeeze(), inp_type.squeeze())

		## Calculate Confusion Matrix
		update_confusion_matrix(AnswerTypeMatrix, YesNoMatrix, StartMatrix, EndMatrix,output, (ans_type, start, end, yes_no))
		if steps%5 == 0 : log_matrices(AnswerTypeMatrix, YesNoMatrix, StartMatrix, EndMatrix, " train", steps)

		## Calculate Loss
		AT_loss = LossFn.loss_AT(output[0], ans_type.squeeze().argmax(1))
		Start_loss = LossFn.loss_start(output[1], start.squeeze().type(torch.FloatTensor))
		End_loss = LossFn.loss_end(output[2], end.squeeze().type(torch.FloatTensor))
		Yes_no_loss = LossFn.loss_yes_no(output[3], yes_no.squeeze())
		
		## Update model params and optim/sched
		total_loss = AT_loss + Start_loss + End_loss + Yes_no_loss
		total_loss.backward()

		## Save loss values
		writer.add_scalars('Loss values',
			{"AT_loss" : AT_loss.item(),"Start_loss":Start_loss.item(), "End_loss":End_loss.item(), "Yes_no_loss":Yes_no_loss.item()},
			steps, time.time())

		cur = time.time() - start
		expected = (cur*num_steps)/steps
		print ("elapsed time : " + str(time.time() - start)+ " : expected time : " +  expected)

		optim.step()
		scheduler.step()     
		
		if steps%20 == 0 : validate(steps/20)


train()













