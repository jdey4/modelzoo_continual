#%%
import numpy as np
import torch
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from random import sample

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pickle
import cv2
import os
# %%
dev = 'gpu'
EPISODES = 100
TRAIN_DATADIR = '/cis/home/jdey4/LargeFineFoodAI/Train'
VAL_DATADIR = '/cis/home/jdey4/LargeFineFoodAI/Val'
SAMPLE_PER_CLASS = 60
NUM_CLASS_PER_TASK = 10
IMG_SIZE = 50
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
food1k =  []

for ii in range(EPISODES):
    food1k.append(
        list(range(ii*10,(ii+1)*10))
    )

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
for tid, task in enumerate(food1k):
    for lid, lab in enumerate(task):
        tmap[lab], cmap[lab] = tid, lid


def get_data(task=0):
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    
    categories_to_consider = range(task*10,(task+1)*10)
    for category in categories_to_consider:
        path = os.path.join(TRAIN_DATADIR, str(category))

        images = os.listdir(path)
        total_images = len(images)
        train_indx = list(range(SAMPLE_PER_CLASS))
        #sample(range(total_images), SAMPLE_PER_CLASS)
        test_indx = np.delete(range(total_images), train_indx)
        for ii in train_indx:
            image_data = cv2.imread(
                    os.path.join(path, images[ii])
                )
            resized_image = cv2.resize(
                image_data, 
                (IMG_SIZE, IMG_SIZE)
            )
            train_X.append(
                resized_image
            )
            train_y.append(
                category
            )
        for ii in test_indx:
            image_data = cv2.imread(
                    os.path.join(path, images[ii])
                )
            resized_image = cv2.resize(
                image_data, 
                (IMG_SIZE, IMG_SIZE)
            )
            test_X.append(
                resized_image
            )
            test_y.append(
                category
            )

    train_X = np.array(train_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    train_y = np.array(train_y)
    test_X = np.array(test_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = np.array(test_y)
    
    return train_X, train_y, test_X, test_y

def create_food1k_task(tasks, train=True, shuffle=False, bs=256):
    # Don't use CIFAR10 mean/std to avoid leaking info 
    # Instead use (mean, std) of (0.5, 0.25)
    transform = transforms.Compose([
        # transforms.Resize(84),
        # transforms.CenterCrop(84),
        transforms.ToTensor(),
    ])
    current_train, current_test = None, None

    cat = lambda x, y: np.concatenate((x, y), axis=0)

    for task_id in tasks:
         data_train, label_train, data_test, label_test = \
         get_data(task=task_id)
         
         if task_id==tasks[0]:
                current_train, current_test = (data_train, label_train), (data_test, label_test)
         else:
                current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
                current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)

    train_dataset = MyDataloader(current_train[0], current_train[1])
    
    test_dataset = MyDataloader(current_test[0], current_test[1])

    #print(dataset.targets[2399], len(dataset))
    train_dataset.targets = [(tmap[train_dataset.targets[j]], cmap[train_dataset.targets[j]]) for j in range(len(train_dataset))]
    test_dataset.targets = [(tmap[test_dataset.targets[j]], cmap[test_dataset.targets[j]]) for j in range(len(test_dataset))]
    # Create dataloader. Set workers to 0, since too few batches
    train_dataloader = DataLoader(
        train_dataset, batch_size=bs, shuffle=shuffle,
        num_workers=0, pin_memory=True)
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=bs, shuffle=shuffle,
        num_workers=0, pin_memory=True)
    
    return train_dataloader, test_dataloader

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
    mttask_loader = create_food1k_task(tasks, shuffle=True, bs=32)
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
    df_singletask['task'] = list(range(1,101))
    df_singletask['accuracy'] = list(zoo_log[ep]['test_acc'])

    with open('food1k/model_zoo.pickle', 'rb') as f:
        pickle.dump(df_singletask, f)

    '''with open('food1k/model_zoo.pickle', 'rb') as f:
        df_multitask = pickle.load(f)

    df_singletask['accuracy'][0] = df_multitask['accuracy'][0]

    summary = (df_multitask, df_singletask)
    with open('food1k/model_zoo.pickle', 'wb') as f:
        pickle.dump(summary, f)'''
    
    return zoo_log
# %%
# Create dataloaders for each task
train_loaders = []
test_loaders = []

for i in range(100):
    tr, te = create_food1k_task([i], train=True)
    train_loaders.append(create_food1k_task([i], train=True))
    test_loaders.append(create_food1k_task([i], train=False))

all_targets = []

for i in range(100):
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
