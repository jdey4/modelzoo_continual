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
EPISODES = 20
#%%
class MyDataloader(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.data = X / 255.
		self.targets = Y

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return torch.from_numpy((self.data[idx] - 0.5) * 2).float(), self.targets[idx]

# %%
mini_imagenet = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
				[10, 11, 12, 13, 14],[15, 16, 17, 18, 19],
				[20, 21, 22, 23, 24], [25, 26, 27, 28, 29],
				[30, 31, 32, 33, 34], [35, 36, 37, 38, 39],
				[40, 41, 42, 43, 44], [45, 46, 47, 48, 49],
				[50, 51, 52, 53, 54], [55, 56, 57, 58, 59],
				[60, 61, 62, 63, 64], [65, 66, 67, 68, 69],
				[70, 71, 72, 73, 74], [75, 76, 77, 78, 79],
				[80, 81, 82, 83, 84], [85, 86, 87, 88, 89],
				[90, 91, 92, 93, 94], [95, 96, 97, 98, 99]]

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
for tid, task in enumerate(mini_imagenet):
	for lid, lab in enumerate(task):
		tmap[lab], cmap[lab] = tid, lid


def create_imagenet_task(tasks, train=True, shuffle=False, bs=256):
	# Don't use CIFAR10 mean/std to avoid leaking info 
	# Instead use (mean, std) of (0.5, 0.25)
	transform = transforms.Compose([
		# transforms.Resize(84),
		# transforms.CenterCrop(84),
		transforms.ToTensor(),
	])
	file_train = open("/Users/jayantadey/TAG/data/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
	file_data_train = pickle.load(file_train)
	data_train = file_data_train["image_data"]

	file_test = open("/Users/jayantadey/TAG/data/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
	file_data_test = pickle.load(file_test)
	data_test = file_data_test["image_data"]

	file_val = open("/Users/jayantadey/TAG/data/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
	file_data_val = pickle.load(file_val)
	data_val = file_data_val["image_data"]

	main_data = data_train.reshape([64, 600, 3, 84, 84])
	test_data = data_test.reshape([20, 600, 3, 84, 84])
	main_data = np.append(main_data, test_data, axis=0)
	val_data = data_val.reshape([16, 600, 3, 84, 84])
	main_data = np.append(main_data, val_data, axis=0)

	all_data = main_data.reshape((60000, 3, 84, 84))
	all_label = np.array([[i] * 600 for i in range(100)]).flatten()

	current_train, current_test = None, None

	cat = lambda x, y: np.concatenate((x, y), axis=0)

	for task_id in tasks:
		for i in mini_imagenet[task_id]:
			class_indices = np.argwhere(all_label == i).reshape(-1)
			class_data = all_data[class_indices]
			class_label = all_label[class_indices]
			split = int(0.8 * class_data.shape[0])

			data_train, data_test = class_data[:split], class_data[split:]
			label_train, label_test = class_label[:split], class_label[split:]

			if i == mini_imagenet[task_id][0] and task_id==tasks[0]:
				current_train, current_test = (data_train, label_train), (data_test, label_test)
			else:
				current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
				current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)
	if train:
		dataset = MyDataloader(current_train[0], current_train[1])
	else:
		dataset = MyDataloader(current_test[0], current_test[1])

	#print(dataset.targets[2399], len(dataset))
	dataset.targets = [(tmap[dataset.targets[j]], cmap[dataset.targets[j]]) for j in range(len(dataset))]
	
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
		self.fc1 = nn.Linear(in_features=254*6*6, out_features=2000)
		self.bn_fc1 = nn.BatchNorm1d(2000)
		self.relu_fc1 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=2000, out_features=2000)
		self.bn_fc2 = nn.BatchNorm1d(2000)
		self.relu_fc2 = nn.ReLU()
		self.fc3 = nn.Linear(in_features=2000, out_features=100)

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
		out = out.view(-1, 254 * 6 * 6)
		
		out = self.fc1(out)
		out = self.bn_fc1(out)
		out = self.relu_fc1(out)
		out = self.fc2(out)
		out = self.bn_fc2(out)
		out = self.relu_fc2(out)
		out = self.fc3(out).reshape(-1,20,5)

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
	mttask_loader = create_imagenet_task(tasks, shuffle=True, bs=32)
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
	df_singletask['task'] = list(range(1,21))
	df_singletask['accuracy'] = list(zoo_log[ep]['test_acc'])

	with open('imagenet/model_zoo.pickle', 'rb') as f:
		df_multitask = pickle.load(f)

	df_singletask['accuracy'][0] = df_multitask['accuracy'][0]

	summary = (df_multitask, df_singletask)
	with open('imagenet/model_zoo.pickle', 'wb') as f:
		pickle.dump(summary, f)
	
	return zoo_log
# %%
# Create dataloaders for each task
train_loaders = []
test_loaders = []

for i in range(20):
	train_loaders.append(create_imagenet_task([i], train=True))
	test_loaders.append(create_imagenet_task([i], train=False))

all_targets = []

for i in range(20):
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
