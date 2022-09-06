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
import tarfile
import os
import cv2
import imageio
# %%
dev = 'cpu'
EPISODES = 5
#%%
class MyDataloader(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.data = X / 255.
		self.targets = torch.from_numpy(Y)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return torch.from_numpy(self.data[idx]).float(), self.targets[idx]

class custom_concat(Dataset):
	r"""
	Subset of a dataset at specified indices.

	Arguments:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
		labels(sequence) : targets as required for the indices. will be the same length as indices
	"""
	def __init__(self, data1, data2):
		#print(data1.data.shape(), data2.data.shape())

		for id in range(len(data2)):
			self.data = np.concatenate((data1.data.reshape(-1,3,32,32),data2[id][0].reshape(-1,3,32,32)), axis=0)
		self.targets = np.concatenate((data1.targets,data2.targets), axis=0)

	def __getitem__(self, idx):
		image = self.data[idx]
		target = self.targets[idx]
		return (image, target)

	def __len__(self):
		return len(self.targets)
# %%
five_dataset = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
				[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
				[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
				[30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
				[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]

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
for tid, task in enumerate(five_dataset):
	for lid, lab in enumerate(task):
		tmap[lab], cmap[lab] = tid, lid

def get_nomnist(task_id):
	"""
	Parses and returns the downloaded notMNIST dataset
	"""
	classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
	tar_path = "./data/notMNIST_small.tar"
	tmp_path = "./data/tmp"

	img_arr = []
	lab_arr = []

	with tarfile.open(tar_path) as tar:
		tar_root = tar.next().name
		for ind, c in enumerate(classes):
			files = [f for f in tar.getmembers() if f.name.startswith(tar_root + '/' + c)]
			if not os.path.exists(tmp_path):
				os.mkdir(tmp_path)
			for f in files:
				f_obj = tar.extractfile(f)
				try:
					arr = np.asarray(imageio.imread(f_obj))
					img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
					img = cv2.resize(img, (32, 32))
					img_arr.append(np.asarray(img))
					lab_arr.append(ind + task_id * len(classes))
				except:
					continue
	os.rmdir(tmp_path)
	return np.array(img_arr), np.array(lab_arr)

def get_5_datasets(task_id, DATA, get_val=False):
	"""
	Returns the data loaders for a single task of 5-dataset
	:param task_id: Current task id
	:param DATA: Dataset class from torchvision
	:param batch_size: Batch size
	:param get_val: Get validation set for grid search
	:return: Train, test and validation data loaders
	"""
	if task_id in [0, 2]:
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),

		])
	else:
		#print('hi', task_id)
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.Resize(32),
			torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
			torchvision.transforms.ToTensor(),
		])
	target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y + task_id * 10)])

	# All datasets except notMNIST (task_id=3) are available in torchvision
	if task_id != 3:
		try:
			train_data = DATA('/Users/jayantadey/progressive-learning-pytorch/data/', train=True, download=True, transform=transforms,
							  target_transform=target_transform)
			test_data = DATA('/Users/jayantadey/progressive-learning-pytorch/data/', train=False, download=True, transform=transforms,
							 target_transform=target_transform)

			#print(train_data[0][0].shape, 'sbrht')
		except:
			# Slighly different way to import SVHN
			train_data = DATA('/Users/jayantadey/progressive-learning-pytorch/data/SVHN/', split='train', download=True, transform=transforms,
							  target_transform=target_transform)
			test_data = DATA('/Users/jayantadey/progressive-learning-pytorch/data/SVHN/', split='test', download=True, transform=transforms,
							 target_transform=target_transform)
		#test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4,
		#                                          pin_memory=True)
	else:
		all_images, all_labels = get_nomnist(task_id)

		#print(all_images.shape, 'task 2 shape')

		dataset_size = len(all_images)
		indices = list(range(dataset_size))
		split = int(np.floor(0.1 * dataset_size))
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]
		train_data = MyDataloader(all_images[train_indices], all_labels[train_indices])
		test_data = MyDataloader(all_images[test_indices], all_labels[test_indices])
		
	return train_data, test_data

def get_5_datasets_tasks(train=True, get_val=False):
	"""
	Returns data loaders for all tasks of 5-dataset.
	:param num_tasks: Total number of tasks
	:param batch_size: Batch-size for training data
	:param get_val: Get validation set for grid search
	"""
	data_list = [torchvision.datasets.CIFAR10,
				 torchvision.datasets.MNIST,
				 torchvision.datasets.SVHN,
				 'notMNIST',
				 torchvision.datasets.FashionMNIST]
	for task_id, DATA in enumerate(data_list):
		print('Loading Task/Dataset:', task_id)
		train_loader, test_loader = get_5_datasets(task_id, DATA)

		if task_id==0:
			if train:
				data_loader = train_loader
			else:
				data_loader = test_loader
		#print(data_loader.data.shape, train_loader.data.shape)
		if train and task_id>0:
			data_loader = custom_concat(data_loader, train_loader)
		elif train!=True and task_id>0:
			data_loader = custom_concat(data_loader, test_loader)

	return data_loader


def create_five_dataset_task(tasks, train=True, shuffle=False, bs=256):
	# Don't use CIFAR10 mean/std to avoid leaking info 
	# Instead use (mean, std) of (0.5, 0.25)
	
	if train:
		path = '/Users/jayantadey/DF-CNN/Data/cifar-100-python/train.pickle'
		with open(path, 'rb') as f:
			df = pickle.load(f)
	else:
		path = '/Users/jayantadey/DF-CNN/Data/cifar-100-python/test.pickle'
		with open(path, 'rb') as f:
			df = pickle.load(f)
	dataset = MyDataloader(df[b'data'].reshape(-1,3,32,32), np.array(df[b'fine_labels']))
	# Select a subset of the data-points, depending on the task
	idx = []
	for task_id in tasks:
		start_class = five_dataset[task_id][0]
		end_class = five_dataset[task_id][9]

		for u in range(start_class, end_class+1):
			idx_ = list(np.where(dataset.targets == u)[0])
			idx.extend(list(idx_))

	#print(int(dataset.targets[0]))
	dataset.targets = [(tmap[int(dataset.targets[j])], cmap[int(dataset.targets[j])]) for j in idx]
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
		self.fc3 = nn.Linear(in_features=2000, out_features=50)

		self.softmax = nn.Softmax()


	def forward(self, x, tasks):
		#print(x.size(0), x.size(1), x.size(2), x.size(3))
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
		#print(out.size(0), out.size(1), out.size(2), out.size(3))
		out = out.view(-1, 254 * 2 * 2)
		
		out = self.fc1(out)
		out = self.bn_fc1(out)
		out = self.relu_fc1(out)
		out = self.fc2(out)
		out = self.bn_fc2(out)
		out = self.relu_fc2(out)
		out = self.fc3(out).reshape(-1,5,10)

		#print(out.size(0), 'fsdsrv')
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

def train_model(tasks, epochs):
	# Train a single model on the Model Zoo
	net = SmallConv()
	if dev != 'cpu':
		net.cuda()
	
	# No cosine-annlealing, fp16, small batch-size
	# This leads to worse accuracies, but leads to simpler/faster code
	mttask_loader = create_five_dataset_task(tasks, shuffle=True, bs=32)
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
def run_zoo(bb=5, epochs=50):
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
		model = train_model(tasks, epochs)
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
	df_singletask['task'] = list(range(1,6))
	df_singletask['accuracy'] = list(zoo_log[ep]['test_acc'])

	with open('five_dataset/model_zoo.pickle', 'rb') as f:
		df_multitask = pickle.load(f)

	df_singletask['accuracy'][0] = df_multitask['accuracy'][0]

	summary = (df_multitask, df_singletask)
	with open('five_dataset/model_zoo.pickle', 'wb') as f:
		pickle.dump(summary, f)
	
	return zoo_log
# %%
# Create dataloaders for each task
train_loaders = []
test_loaders = []

for i in range(5):
	train_loaders.append(create_five_dataset_task([i], train=True))
	test_loaders.append(create_five_dataset_task([i], train=False))

all_targets = []

for i in range(5):
	all_targets.append([])
	for loader in [train_loaders[i], test_loaders[i]]:
		task_targets = []
		for dat, target in loader:
			task_targets.append(target[1].numpy())
		task_targets = np.concatenate(task_targets)
		all_targets[i].append(task_targets)

#zoo_log = run_zoo(bb=5, epochs=5)
isolated_log = run_zoo(bb=1, epochs=5)
# %%
