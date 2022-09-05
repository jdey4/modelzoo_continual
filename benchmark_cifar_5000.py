#%%
import numpy as np
import torch
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pickle
# %%
dev = 'cpu'
EPISODES = 10
# %%
# Download dataset
_ = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
_ = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

coarse_cifar100 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
				   [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
				   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
				   [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
				   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
				   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
				   [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
				   [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
				   [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
				   [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]

'''task_names = ["Aq. Mammals", "Fish",
			  "Flowers", "Food Container",
			  "Fruits and Veggies", "Electrical Devices",
			  "Household Furniture", "Insects", 
			  "Large Carnivores", "Man-made Outdoor",
			  "Natural Outdoor", "Omni-Herbivores",
			  "Med. Mammals", "Invertebrates",
			  "People", "Reptiles",
			  "Small Mammals", "Trees",
			  "Vehicles 1", "Vehicles 2"]'''

# Map each cifar100 label -> (task, task-label)
tmap, cmap = {}, {}
for tid, task in enumerate(coarse_cifar100):
	for lid, lab in enumerate(task):
		tmap[lab], cmap[lab] = tid, lid

class custom_concat(Dataset):
	r"""
	Subset of a dataset at specified indices.

	Arguments:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
		labels(sequence) : targets as required for the indices. will be the same length as indices
	"""
	def __init__(self, data1, data2):
		self.data = np.concatenate((data1.data.reshape(-1,3,32,32),data2.data.reshape(-1,3,32,32)), axis=0)
		self.targets = np.concatenate((data1.targets,data2.targets), axis=0)

	def __getitem__(self, idx):
		image = self.data[idx]
		target = self.targets[idx]
		return (image, target)

	def __len__(self):
		return len(self.targets)

def create_cifar100_task(tasks, slot, shift, train=True, shuffle=False, bs=256):
	# Don't use CIFAR10 mean/std to avoid leaking info 
	# Instead use (mean, std) of (0.5, 0.25)
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
	])
	dataset_train = torchvision.datasets.CIFAR100(
		root='./data', train=True, download=False, transform=transform)
	
	dataset_test = torchvision.datasets.CIFAR100(
		root='./data', train=False, download=False, transform=transform)

	dataset = custom_concat(dataset_train, dataset_test)
	# Select a subset of the data-points, depending on the task
	train_idx = []
	test_idx= []
	for task_id in tasks:
		start_class = coarse_cifar100[task_id][0]
		end_class = coarse_cifar100[task_id][9]
		idx = [np.where(dataset.targets == u)[0] for u in range(start_class, end_class+1)]
		
		for cls in range(end_class-start_class+1):
			indx = np.roll(idx[cls],(shift-1)*100)
			#print(combined_targets[indx[0]])
			#print(indx[0][slot*50:(slot+1)*50], slot)
			train_idx.extend(list(indx[slot*50:(slot+1)*50]))
			test_idx.extend(list(indx[500:600]))
	
	if train:
		idx = train_idx
	else:
		idx = test_idx

	dataset.targets = [(tmap[dataset.targets[j]], cmap[dataset.targets[j]]) for j in idx]
	dataset.data = dataset.data[idx]
	
	# Create dataloader. Set workers to 0, since too few batches
	dataloader = DataLoader(
		dataset, batch_size=bs, shuffle=shuffle,
		num_workers=0, pin_memory=True)
	
	return dataloader

#%%
class SmallConv(nn.Module):
	# Small convolution network
	def __init__(self, channels=3, avg_pool=2, lin_size=320):
		super(SmallConv, self).__init__()
		
		self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu1 = nn.ReLU()
		self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)
		self.relu2 = nn.ReLU()
		self.layer3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.layer4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(128)
		self.relu4 = nn.ReLU()
		self.layer5 = nn.Conv2d(128, 254, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(254)
		self.relu5 = nn.ReLU()
		self.fc1 = nn.Linear(in_features=254*2*2, out_features=2000)
		self.bn_fc1 = nn.BatchNorm1d(2000)
		self.relu_fc1 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=2000, out_features=2000)
		self.bn_fc2 = nn.BatchNorm1d(2000)
		self.relu_fc2 = nn.ReLU()
		self.fc3 = nn.Linear(in_features=2000, out_features=100)

		self.softmax = nn.Softmax()


	def forward(self, x, tasks):

		out = self.layer1(x.float())
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.layer2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.layer3(out)
		out = self.bn3(out)
		out = self.relu3(out)
		out = self.layer4(out)
		out = self.bn4(out)
		out = self.relu4(out)
		out = self.layer5(out)
		out = self.bn5(out)
		out = self.relu5(out)
		out = out.view(-1, 254 * 2 * 2)
		out = self.fc1(out)
		out = self.bn_fc1(out)
		out = self.relu_fc1(out)
		out = self.fc2(out)
		out = self.bn_fc2(out)
		out = self.relu_fc2(out)
		out = self.fc3(out).reshape(-1,10,10)
		out = out[torch.arange(out.size(0)), list(tasks), :]
		out = self.softmax(out)
		
		return out
# %%
def process_dat(dat, target):
	# Helper function to push data to gpu
	dat = dat.to(dev)
	tasks, labels = target
	labels = labels.long().to(dev)
	tasks = tasks.long().to(dev)
	return dat, labels, tasks


def sample_task(train_losses, episode, bb):
	# Samples "bb" tasks to train on
	bb = min(bb, episode+1)

	if episode == 0:
		return [0]

	prob = (train_losses - np.mean(train_losses)) / np.mean(train_losses)
	prob = np.exp(prob).clip(0.0001, 1000)
	prob = prob / np.sum(prob)

	tasks = list(np.random.choice(episode, bb-1, replace=False, p=prob))
	tasks.append(episode) # Add the newest task to the list
	tasks = [int(t) for t in tasks]
	return tasks

def train_model(tasks, epochs, slot, shift):
	# Train a single model on the Model Zoo
	net = SmallConv()
	if dev != 'cpu':
		net.cuda()
	
	# No cosine-annlealing, fp16, small batch-size
	# This leads to worse accuracies, but leads to simpler/faster code
	mttask_loader = create_cifar100_task(tasks, slot, shift, shuffle=True, bs=32)
	optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9,
						  nesterov=True,weight_decay=0.00001)
	criterion = nn.CrossEntropyLoss()

	net.train()
	for ep in tqdm(range(epochs)):
		for dat, target in mttask_loader:
			optimizer.zero_grad()
			dat, labels, tasks = process_dat(dat, target)
			out = net(dat.float(), tasks)
			loss = criterion(out, labels)
			loss.backward()
			optimizer.step()
	return net


def update_zoo(net, tasks, zoo_outputs):
	# Instead of storing the new network into the model zoo, we store all
	# outputs of the network for faster run-times
	criterion = nn.CrossEntropyLoss()
	net.eval()

	for tid in tasks:
		for idx, dataloader in enumerate([train_loaders[tid], test_loaders[tid]]):
			outputs = []
			for dat, target in dataloader:
				dat, labels, tasks = process_dat(dat, target)
				out = net(dat.float(), tasks)
				out = nn.functional.softmax(out, dim=1)
				out = out.cpu().detach().numpy()
				outputs.append(out)
			outputs = np.concatenate(outputs)
			zoo_outputs[tid][idx].append(outputs)


def evaluate_zoo(episode, zoo_outputs):
	# Evaluate the entire model zoo and all tasks seen until current episode
	criterion = nn.NLLLoss()
	met = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc":[]}

	for tid in range(episode+1):
		train_preds = np.mean(zoo_outputs[tid][0], axis=0)
		test_preds = np.mean(zoo_outputs[tid][1], axis=0)

		tr_loss = criterion(torch.Tensor(train_preds).log(),
							torch.Tensor(all_targets[tid][0]).long())
		te_loss = criterion(torch.Tensor(test_preds).log(),
							torch.Tensor(all_targets[tid][1]).long())
 
		met["train_loss"].append(tr_loss.item())
		met["test_loss"].append(te_loss.item())
		met["train_acc"].append((train_preds.argmax(1) == all_targets[tid][0]).mean())
		met["test_acc"].append((test_preds.argmax(1) == all_targets[tid][1]).mean())

	return met
# %%
def run_zoo(slot, shift, bb=5, epochs=50):
	zoo_outputs = {}
	zoo_log = {}
	train_losses = []

	'''tasks_ = []
	base_tasks = []
	accuracies_across_tasks = []
	df_multitask = pd.DataFrame()'''
	singletask_accuracy = []
	df_singletask = pd.DataFrame()

	for ep in range(EPISODES):
		'''base_tasks.extend([ep+1]*(ep+1))
		tasks_.extend(list(range(1,ep+2)))'''

		print("Epsisode " + str(ep))
		zoo_outputs[ep] = [[], []]  
		tasks = sample_task(train_losses, ep, bb)
	
		print("Training model on tasks: " + str(tasks))
		model = train_model(tasks, epochs, slot, shift)
		update_zoo(model, tasks, zoo_outputs)
		zoo_log[ep] = evaluate_zoo(ep, zoo_outputs)
		train_losses = zoo_log[ep]["train_loss"]

		#accuracies_across_tasks.extend(list(zoo_log[ep]['test_acc']))
		print("Test Accuracies of the zoo:\n  %s\n" % str(zoo_log[ep]['test_acc']))

	#print(tasks_, 'tasks')
	#print(base_tasks, 'base')
	#print(accuracies_across_tasks, 'acc')
	'''df_multitask['task'] = tasks_
	df_multitask['base_task'] = base_tasks
	df_multitask['accuracy'] = accuracies_across_tasks'''
	df_singletask['task'] = list(range(1,11))
	df_singletask['accuracy'] = list(zoo_log[ep]['test_acc'])

	with open('results/model_zoo_'+ str(slot+1)+'_'+str(shift)+'.pickle', 'rb') as f:
		df_multitask = pickle.load(f)

	df_singletask['accuracy'][0] = df_multitask['accuracy'][0]

	summary = (df_multitask, df_singletask)
	with open('results/model_zoo_'+ str(slot+1)+'_'+str(shift)+'.pickle', 'wb') as f:
		pickle.dump(summary, f)
	
	return zoo_log
# %%
for shift in range(1,7):
	for slot in range(10):
		# Create dataloaders for each task
		train_loaders = []
		test_loaders = []

		for i in range(10):
			train_loaders.append(create_cifar100_task([i], slot=slot, shift=shift, train=True))
			test_loaders.append(create_cifar100_task([i], slot=slot, shift=shift, train=False))

		all_targets = []

		for i in range(10):
			all_targets.append([])
			for loader in [train_loaders[i], test_loaders[i]]:
				task_targets = []
				for dat, target in loader:
					task_targets.append(target[1].numpy())
				task_targets = np.concatenate(task_targets)
				all_targets[i].append(task_targets)

		#zoo_log = run_zoo(slot, shift, bb=5, epochs=5)
		isolated_log = run_zoo(slot, shift, bb=1, epochs=5)
# %%
