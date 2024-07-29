import numpy as np
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

import pdb
import random
import torch
import time
import cv2


data_path = "../224kfold/"


class rtPACS(Dataset):
    def __init__(self, test_domain, num_samples=20, num_domains=3, transform=None):
        self.domain_list = ["art_painting", "photo", "cartoon", "sketch"]
        self.domain_list.remove(test_domain)
        self.num_domains = num_domains
        self.num_samples = num_samples
        assert self.num_domains <= len(self.domain_list)

        self.sample_list = []

        self.infer_imgs = []
        self.infer_labels = []

        for i in range(len(self.domain_list)):
            f = open("../files/" + self.domain_list[i] + "_train_kfold.txt", "r")
            lines = f.readlines()
            samples = {}

            for line in lines:
                [img, label] = line.strip("\n").split(" ")
                label = int(label) - 1
                if label not in samples.keys():
                    samples[label] = []
                samples[label].append(data_path + img)
            self.sample_list.append(samples)

            for i in range(len(samples.keys())):
                self.infer_imgs = self.infer_imgs + samples[i][:num_samples]
                self.infer_labels = self.infer_labels + [i] * num_samples

    def reset(self, phase, domain_id, transform=None):
        self.phase = phase
        self.transform = transform
        if phase == "train":
            self.img_list = []
            self.label_list = []
            for i in range(self.num_domains):
                if i == domain_id:
                    continue
                for j in range(7):
                    np.random.shuffle(self.sample_list[i][j])
                    self.img_list = (
                        self.img_list + self.sample_list[i][j][: self.num_samples]
                    )
                    self.label_list = self.label_list + [j] * self.num_samples
        else:
            self.img_list = self.infer_imgs
            self.label_list = self.infer_labels
        assert len(self.img_list) == len(self.label_list)

    def __getitem__(self, item):
        image = Image.open(self.img_list[item]).convert("RGB")
        img_name = self.img_list[item]
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_list[item]

        return image, label, img_name

    def __len__(self):
        return len(self.img_list)

