import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier

from torchvision.datasets import CIFAR10, CIFAR100
from datasets import TinyImageNet

class FeatureDataset(Dataset):
    def __init__(self, dataset):
        self.set = dataset
        self.feature = None
        if isinstance(self.set, CIFAR10) or isinstance(self.set, CIFAR100):
            self.gt = np.array(self.set.targets)
        elif isinstance(self.set, TinyImageNet):
            self.gt = np.array([label for _, label in self.set.data])
        else:
            raise NotImplementedError

    def __len__(self):
        return self.set.__len__()

    def __getitem__(self, idx):
        return self.set.__getitem__(idx)

    def set_feature(self, z):
        self.feature = z.astype(float)


class KNN():
    def __init__(self, train_set, test_set, batch_size=128):
        self.train_set = FeatureDataset(train_set)
        self.test_set = FeatureDataset(test_set)
        self.batch_size = batch_size

    def evaluate(self, feat_func):
        self.update_features(self.train_set, feat_func)
        self.update_features(self.test_set, feat_func)

        neigh = KNeighborsClassifier(n_neighbors=20)
        neigh.fit(self.train_set.feature, self.train_set.gt)
        test_knn = neigh.score(self.test_set.feature, self.test_set.gt)
        print("test K-NN acc =", test_knn)
        return test_knn

    def update_features(self, which_set, feat_func):
        ''' Extract features for all data samples through feat_func. '''
        dataloader = DataLoader(
            which_set,
            batch_size=self.batch_size,
            shuffle=False, # important!
        )
        outputs = []
        for x_batch, *_ in tqdm(dataloader):
            with torch.no_grad():
                outputs.append(feat_func(x_batch.cuda()).detach().cpu())
        result = torch.cat(outputs, dim=0).numpy()
        which_set.set_feature(result)


class LinearProbe():
    def __init__(self, train_set, test_set, num_classes, batch_size=128, lr=1e-3, epoch=15):
        self.train_set = train_set
        self.test_set = test_set
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, feat_func):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        hidden_size = feat_func(next(iter(test_loader))[0].cuda()).shape[-1]
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6),
            nn.Linear(hidden_size, self.num_classes),
        ).cuda()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optim, self.epoch)

        # training
        for e in range(self.epoch):
            pbar = tqdm(train_loader)
            for x, y in pbar:
                pbar.set_description("[epoch %d]: lr: %.1e" % (e, self.optim.param_groups[0]['lr']))
                self.classifier.train()
                with torch.no_grad():
                    feat = feat_func(x.cuda()).detach().float()
                logit = self.classifier(feat)
                loss = self.loss_fn(logit, y.cuda())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            self.scheduler.step()

        # testing
        preds = []
        labels = []
        for x, y in tqdm(test_loader):
            with torch.no_grad():
                self.classifier.eval()
                feat = feat_func(x.cuda()).detach().float()
                logit = self.classifier(feat)
                pred = logit.argmax(dim=-1)
                preds.append(pred)
                labels.append(y)
        pred = torch.cat(preds)
        label = torch.cat(labels)
        acc = (pred.cpu() == label).sum().item() / len(label)
        print("test Linear Probe acc =", acc)
        return acc
