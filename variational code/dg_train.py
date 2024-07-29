from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import os, shutil
import argparse
import time

from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from aug import *
import pdb
from pacs_rtdataset import *
from pacs_dataset import *


import dg_maml_vi_model
import sys
import numpy as np

from torch.nn import init
from sklearn.model_selection import train_test_split

bird = False
import psutil

cpu_workers = 4

from timm.loss import JsdCrossEntropy
from math import remainder

import learn2learn as l2l

from learn2learn.data.transforms import (
    NWays,
    KShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)

import math

import pdb

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="learning rate")
parser.add_argument("--sparse", default=0, type=float, help="L1 panelty")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)

parser.add_argument("--log_dir", default="log1", help="Log dir [default: log]")
parser.add_argument("--dataset", default="PACS", help="datasets")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch Size during training [default: 32]",
)
parser.add_argument(
    "--pseudo_label_update_epoch",
    default=10,
    type=int,
    help="epoch to update pseudo labels",
)

parser.add_argument(
    "--shuffle", type=int, default=0, help="Batch Size during training [default: 32]"
)
parser.add_argument(
    "--optimizer", default="adam", help="adam or momentum [default: adam]"
)

parser.add_argument("--net", default="res18", help="res18 or res50")


parser.add_argument("--autodecay", action="store_true")


parser.add_argument(
    "--test_domain", default="photo", help="GPU to use [default: GPU 0]"
)
parser.add_argument("--train_domain", default="", help="GPU to use [default: GPU 0]")
parser.add_argument("--ite_train", default=True, type=bool, help="learning rate")
parser.add_argument("--max_ite", default=10000, type=int, help="max_ite")
parser.add_argument("--test_ite", default=50, type=int, help="learning rate")
parser.add_argument("--bias", default=1, type=int, help="whether sample")
parser.add_argument("--test_batch", default=100, type=int, help="learning rate")
parser.add_argument("--data_aug", default=1, type=int, help="whether sample")
parser.add_argument("--difflr", default=1, type=int, help="whether sample")


parser.add_argument("--reslr", default=0.5, type=float, help="backbone learning rate")

parser.add_argument("--agg_model", default="concat", help="concat or bayes or rank1")
parser.add_argument("--agg_method", default="mean", help="ensemble or mean or ronly")


parser.add_argument("--ctx_num", default=10, type=int, help="learning rate")
parser.add_argument("--hierar", default=2, type=int, help="hierarchical model")


parser.add_argument(
    "--model_saving_dir",
    default="./models_new/models_code",
    type=str,
    help=" place to save the best model obtained during training",
)

parser.add_argument(
    "--resume_from_checkpoint", type=str, help=" resume from checkpoint"
)

parser.add_argument("--num_iterations", default=1000, type=int, help="learning rate")
parser.add_argument(
    "--entropy_threshold_perc", default=0.9, type=float, help="entropy_threshold_perc"
)
parser.add_argument(
    "--lr_adam_maml", default=0.01, type=float, help="learning rate Maml"
)


args = parser.parse_args()


BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer

backbone = args.net

max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
iteration_training = args.ite_train

test_domain = args.test_domain
train_domain = args.train_domain


ctx_num = args.ctx_num


difflr = args.difflr
res_lr = args.reslr
hierar = args.hierar
agg_model = args.agg_model


with_bias = args.bias
with_bias = bool(with_bias)
difflr = bool(difflr)
pseudo_label_update_epoch = args.pseudo_label_update_epoch


data_aug = args.data_aug
data_aug = bool(data_aug)
model_saving_dir = args.model_saving_dir
resume_from_checkpoint = args.resume_from_checkpoint


run_name = "./"

LOG_DIR = os.path.join("logs_meta", "")
MODEL_DIR = os.path.join(model_saving_dir, "")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(os.path.join(LOG_DIR, "validation")):
    os.makedirs(os.path.join(LOG_DIR, "validation"))

if not os.path.exists(os.path.join(LOG_DIR, "test")):
    os.makedirs(os.path.join(LOG_DIR, "test"))


if not os.path.exists(os.path.join(LOG_DIR, "logs")):
    os.makedirs(os.path.join(LOG_DIR, "logs"))
text_file = os.path.join(LOG_DIR, "log_train.txt")
text_file2 = os.path.join(LOG_DIR, "log_std_output.txt")


import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(text_file2, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()

LOG_FOUT = open(text_file, "w")

print(args)
LOG_FOUT.write(str(args) + "\n")


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    if print_out:
        print(out_str)


tr_writer = SummaryWriter(LOG_DIR)
val_writer = SummaryWriter(os.path.join(LOG_DIR, "validation"))
te_writer = SummaryWriter(os.path.join(LOG_DIR, "test"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


best_acc = 0
best_valid_acc = 0
start_epoch = 0


decay_inter = [250, 450]


print("==> Preparing data..")

if args.dataset == "PACS":
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0

    ctx_test = ctx_num
    domains = ["art_painting", "photo", "cartoon", "sketch"]
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if data_aug:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    all_dataset = PACS(test_domain)
    rt_context = rtPACS(test_domain, ctx_num)
else:
    raise NotImplementedError




args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird


print("--> --> LOG_DIR <-- <--", LOG_DIR)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


net = dg_maml_vi_model.ResNet18_vi()

print("==> Building model..")
print(net)


net.apply(inplace_relu)

net = net.to(device)




net.train()


if device == "cuda":
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay


NUM_IMAGES_PER_BATCH = 20



def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(data, labels, learner, loss, adaptation_steps, shots, ways, device):
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()

    if data is not None:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(data, labels, test_size=0.5, stratify=labels)
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        for step in range(adaptation_steps):
            preds, _ = learner(adaptation_data)
            adaptation_error = loss(preds, adaptation_labels)
            learner.adapt(adaptation_error)

        predictions, _ = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)

        return evaluation_error, evaluation_accuracy


def fast_adapt_pl(data, labels, learner, loss, adaptation_steps, shots, ways, device):
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()

    try:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(data, labels, test_size=0.5, stratify=labels)
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        for step in range(adaptation_steps):
            predictions = learner(adaptation_data)
            pseudo_label = torch.argmax(predictions, dim=1)
            pseudo_label = find_neighbors(pseudo_label, adaptation_labels)
            pseudo_label = torch.from_numpy(pseudo_label).long().to(device)
            pseudo_label = pseudo_label.to(device)

            w_loss = wasserstein_distance_torch(pseudo_label, adaptation_labels)

            adaptation_error = loss(learner(adaptation_data), pseudo_label)
            learner.adapt(adaptation_error)

        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)

        return evaluation_error, w_loss, evaluation_accuracy
    except:
        log_string("Error in fast_adapt, train test split")
        val = 0
        return (
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
        )


nll_loss = torch.nn.NLLLoss()


nll_loss = torch.nn.NLLLoss()


def alpha_weight_for_pseudo_label_loss(iter):
    alpha = 1 / (1 + np.exp(-iter / 1000))
    return alpha


def fast_adapt_pl_entropy(
    data, labels, learner, loss, adaptation_steps, shots, ways, device, iter
):
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()

    try:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(data, labels, test_size=0.5, stratify=labels)
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        for step in range(adaptation_steps):
            adaptation_error = loss(learner(adaptation_data), adaptation_labels)
            w_loss = wasserstein_distance_torch(adaptation_labels, adaptation_labels)
            learner.adapt(adaptation_error)

        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)

        _, pseudo_label = torch.max(predictions, 1)

        alpha = alpha_weight_for_pseudo_label_loss(iter)
        alpha = alpha + 0.5
        pseudo_label_error = alpha * (loss(predictions, pseudo_label))

        pseudo_label_accuracy = 0.5

        actual_loss = loss(predictions, evaluation_labels)

        return (
            pseudo_label_error,
            pseudo_label_accuracy,
            evaluation_error,
            w_loss,
            evaluation_accuracy,
            actual_loss,
        )
    except:
        log_string("Error in fast_adapt, train test split")
        val = 0
        return (
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
        )


ul_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


def filter_predictions_based_on_entropy(predictions, entropy_threshold=0.5):
    entropy_threshold = 0.2

    entropy = torch.nn.functional.softmax(predictions, dim=1)

    print("entropy size", entropy.shape)
    entropy_threshold_indices = entropy < entropy_threshold

    print("True False", entropy_threshold_indices)

    entropy_threshold_indices = entropy_threshold_indices.sum(dim=1) > 4
    print("entropy size", entropy.shape)
    entropy_threshold_indices = torch.where(entropy_threshold_indices)
    print("torch where", entropy_threshold_indices)

    filtered_predictions = predictions[entropy_threshold_indices]
    return filtered_predictions, entropy_threshold_indices


def filter_data_and_labels_one_occurence(data, labels):
    l1 = []
    for i in range(len(labels)):
        if np.sum(labels == labels[i]) == 1:
            l1.append(i)

    data = np.delete(data, l1, axis=0)
    labels = np.delete(labels, l1, axis=0)
    l1_indices = l1
    return data, labels, l1_indices


def filter_data_and_labels_two_occurence(data, labels):
    unique_labels = np.unique(labels)
    indices_to_delete = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 2:
            indices_to_delete.extend(indices[2:])

    labels = np.delete(labels, indices_to_delete)

    data = np.delete(data, indices_to_delete, axis=0)
    return data, labels


def kl_divergence_vi(mu_q, sigma_q, mu_p, sigma_p):
    var_q = sigma_q**2 + 1e-6
    var_p = sigma_p**2 + 1e-6
    component1 = torch.log(var_p) - torch.log(var_q)
    component2 = var_q / var_p
    component3 = (mu_p - mu_q).pow(2) / var_p
    KLD = 0.5 * torch.sum((component1 - 1 + component2 + component3), 1)

    return KLD


def fast_adapt_pl_entropy_new_vi(
    data, labels_actual, learner, loss, adaptation_steps, shots, ways, device, iter
):
    labels, _ = learner(data)

    data = data.cpu().numpy()
    data_actual = data.copy()

    labels = torch.argmax(labels, dim=1)
    labels = labels.cpu().numpy()

    data, labels, l1_indices = filter_data_and_labels_one_occurence(data, labels)

    labels_actual = labels_actual.cpu().numpy()
    data_actual = np.delete(data_actual, l1_indices, axis=0)
    labels_actual = np.delete(labels_actual, l1_indices, axis=0)

    kl_coeff = 0.05

    if data is not None:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(
            data, labels, test_size=0.5, stratify=labels, random_state=42
        )
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        (
            actual_data_train,
            actual_data_test,
            actual_labels_train,
            actual_labels_test,
        ) = train_test_split(
            data_actual,
            labels_actual,
            test_size=0.5,
            stratify=labels_actual,
            random_state=42,
        )
        actual_data_train, actual_labels_train = torch.from_numpy(actual_data_train).to(
            device
        ), torch.from_numpy(actual_labels_train).to(device)
        actual_data_test, actual_labels_test = torch.from_numpy(actual_data_test).to(
            device
        ), torch.from_numpy(actual_labels_test).to(device)

        for step in range(adaptation_steps):
            _, features = learner(adaptation_data)
            pDz = []
            for cate in range(7):
                if cate in actual_labels_train.unique():
                    pDz.append(
                        features[actual_labels_train == cate].mean(0, keepdim=True)
                    )

                else:
                    pDz.append(features.mean(0, keepdim=True))

            pDz = torch.cat(pDz, 0)
            pDz = pDz.detach()

            pw_mu, pw_sigma = learner.module.classifier(pDz)

            qDz = []
            for cate in range(7):
                if cate in adaptation_labels.unique():
                    qDz.append(
                        features[adaptation_labels == cate].mean(0, keepdim=True)
                    )

                else:
                    qDz.append(features.mean(0, keepdim=True))
            qDz = torch.cat(qDz, 0)
            qDz = qDz.detach()
            qw_mu, qw_sigma = learner.module.classifier(qDz)

            kld = kl_divergence_vi(qw_mu, qw_sigma, pw_mu, pw_sigma)

            mc_times = 10
            qmu_samp = qw_mu.unsqueeze(1).repeat(1, mc_times, 1)
            qsigma_samp = qw_sigma.unsqueeze(1).repeat(1, mc_times, 1)

            eps_q = qmu_samp.new(qmu_samp.size()).normal_()
            qw = qmu_samp + 1 * qsigma_samp * eps_q
            y = torch.mm(
                features.detach(),
                qw.permute(2, 1, 0).contiguous().view(512, mc_times * 7),
            )
            y = y.view(len(adaptation_labels), mc_times, 7).mean(1)

            refine_error = loss(y, actual_labels_train)

            kld = kld.sum()
            kld = kld * kl_coeff
            learner.adapt(refine_error)
            learner.adapt(kld)

            _, updated_pseudo_label = torch.max(y, 1)

            preds, _ = learner(adaptation_data)
            adaptation_error = loss(preds, adaptation_labels)

            if iter < 60:
                gamma = 0.01
            else:
                gamma = 1
            classifier_gen_error = adaptation_error * gamma

            learner.adapt(adaptation_error)
            learner.adapt(classifier_gen_error)

        _, features = learner(evaluation_data)
        pDz = []
        for cate in range(7):
            if cate in actual_labels_test.unique():
                pDz.append(features[actual_labels_test == cate].mean(0, keepdim=True))

            else:
                pDz.append(features.mean(0, keepdim=True))

        pDz = torch.cat(pDz, 0)

        pw_mu, pw_sigma = learner.module.classifier(pDz)

        qDz = []
        for cate in range(7):
            if cate in evaluation_labels.unique():
                qDz.append(features[evaluation_labels == cate].mean(0, keepdim=True))

            else:
                qDz.append(features.mean(0, keepdim=True))
        qDz = torch.cat(qDz, 0)
        qw_mu, qw_sigma = learner.module.classifier(qDz)

        kld = kl_divergence_vi(qw_mu, qw_sigma, pw_mu, pw_sigma)

        mc_times = 10
        qmu_samp = qw_mu.unsqueeze(1).repeat(1, mc_times, 1)
        qsigma_samp = qw_sigma.unsqueeze(1).repeat(1, mc_times, 1)

        eps_q = qmu_samp.new(qmu_samp.size()).normal_()
        qw = qmu_samp + 1 * qsigma_samp * eps_q
        y = torch.mm(features, qw.permute(2, 1, 0).contiguous().view(512, mc_times * 7))
        y = y.view(len(evaluation_labels), mc_times, 7).mean(1)

        refine_error = loss(y, actual_labels_test)

        predictions, _ = learner(evaluation_data)
        _, pseudo_label = torch.max(predictions, 1)
        _, updated_pseudo_label = torch.max(y, 1)

        alpha = 0.9

        pseudo_label_error = alpha * (loss(predictions, pseudo_label))

        actual_accuracy = accuracy(learner(actual_data_test)[0], actual_labels_test)

        beta = 0.1
        actual_loss = beta * (loss(learner(actual_data_test)[0], actual_labels_test))

        return (
            pseudo_label_error,
            actual_accuracy,
            actual_loss,
            alpha,
            beta,
            refine_error,
            kld,
            classifier_gen_error,
        )


def fast_adapt_test_time(
    data, learner, loss, adaptation_steps, shots, ways, device, iter
):
    labels = learner(data)
    data = data.cpu().numpy()

    labels = torch.argmax(labels, dim=1)
    labels = labels.cpu().numpy()

    data, labels = filter_data_and_labels_one_occurence(data, labels)

    data, labels = filter_data_and_labels_two_occurence(data, labels)

    if data is not None:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(data, labels, test_size=0.5, stratify=labels)
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        adaptation_steps = 1
        for step in range(adaptation_steps):
            predictions = learner(adaptation_data)
            _, pseudo_label = torch.max(predictions, 1)

            adaptation_error = loss(predictions, pseudo_label)

            learner.adapt(adaptation_error)

        predictions = learner(evaluation_data)

        _, pseudo_label = torch.max(predictions, 1)
        alpha = alpha_weight_for_pseudo_label_loss(iter)
        alpha = alpha + 0.4
        pseudo_label_error = 0.00001 * (loss(predictions, pseudo_label))

        evaluation_accuracy = 0.5

        return pseudo_label_error, evaluation_accuracy


def train_meta(args):
    python_file_name = os.path.basename(__file__)

    current_directory = os.getcwd()
    python_file_name = os.path.join(current_directory, python_file_name)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    kl_loss_tot = 0
    w_loss_tot = 0
    js_div_tot = 0
    correct_source = 0
    total_source = 0
    adapt_loss_tot = 0

    batch_size = 32

    adaptation_steps = 1
    num_iterations = 20000
    shots = 5
    ways = 7
    fast_lr = 0.5
    kl_coeff = 0.05
    num_iterations = args.num_iterations

    lr_adam_maml = args.lr_adam_maml
    maml = l2l.algorithms.MAML(
        net, lr=lr_adam_maml, first_order=False, allow_nograd=True, allow_unused=True
    )
    maml.to(device)
    print(maml)

    optimizer = torch.optim.Adam(
        [
            {"params": maml.module.features.parameters(), "lr": args.lr * res_lr},
            {
                "params": maml.module.fc.parameters(),
            },
            {
                "params": maml.module.classifier.parameters(),
            },
        ],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    scheduler_lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.01,
        eps=1e-08,
    )

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    checkpoint_dir = os.path.join(MODEL_DIR, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_valid_acc = 0

    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            log_string("Loading checkpoint from {}".format(args.resume_from_checkpoint))
            checkpoint = torch.load(args.resume_from_checkpoint)
            maml.load_state_dict(checkpoint["net"])

            best_valid_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]
            log_string("Loaded checkpoint from epoch {}".format(start_epoch))
            log_string("\n \n ===> Loaded model from checkpoint <==========")
        else:
            log_string("No checkpoint found at {}".format(args.resume_from_checkpoint))
            exit()

    for iter in range(num_iterations):
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        t0 = time.time()

        domain_id = np.random.randint(len(domains))

        print("Domain", domain_id)
        all_dataset.reset("train", domain_id, transform=transform_train)

        kl_loss_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        rt_context.reset("train", domain_id, transform=transform_train)
        train_dataset = l2l.data.MetaDataset(rt_context)
        transforms = [
            l2l.data.transforms.NWays(train_dataset, ways),
            l2l.data.transforms.KShots(train_dataset, 2 * shots),
            l2l.data.transforms.LoadData(train_dataset),
        ]
        taskset = l2l.data.TaskDataset(train_dataset, transforms, num_tasks=1000)

        val_dataset = l2l.data.MetaDataset(all_dataset)
        transforms_val = [
            l2l.data.transforms.NWays(val_dataset, ways),
            l2l.data.transforms.KShots(val_dataset, 2 * shots),
            l2l.data.transforms.LoadData(val_dataset),
        ]
        taskset_val = l2l.data.TaskDataset(val_dataset, transforms_val, num_tasks=1000)

        for counter, (inputs, targets, img_name2) in enumerate(taskset):
            learner = maml.clone()

            data, labels = inputs.to(device), targets.to(device)

            eval_error_tr, eval_accuracy_tr = fast_adapt(
                data, labels, learner, loss, adaptation_steps, shots, ways, device
            )
            eval_error_tr.backward()
            meta_train_error += eval_error_tr.item()
            meta_train_accuracy += eval_accuracy_tr.item()
            if counter == (batch_size - 1):
                break

        log_string(
            "Iteration %d: train error %.2f, train accuracy %.2f"
            % (iter, meta_train_error / batch_size, meta_train_accuracy / batch_size)
        )

        for p in maml.module.features.parameters():
            p.grad.data.mul_(1.0 / batch_size)
        for p in maml.module.fc.parameters():
            p.grad.data.mul_(1.0 / batch_size)
        optimizer.step()

        optimizer.zero_grad()

        counter_val = 0
        ws_loss_it = 0
        pseudo_error_val_tot = 0
        refine_error_val_tot = 0
        kl_div_val_tot = 0
        classifier_gen_error_tot = 0
        if math.remainder(iter, 1) == 0:
            for counter, (inputs, targets, img_name2) in enumerate(taskset_val):
                learner = maml.clone()

                data, labels = inputs.to(device), targets.to(device)

                (
                    pseudo_label_error,
                    eval_accuracy_val,
                    eval_error_val,
                    alpha,
                    beta,
                    refine_error,
                    kld,
                    classifer_gen_error,
                ) = fast_adapt_pl_entropy_new_vi(
                    data,
                    labels,
                    learner,
                    loss,
                    adaptation_steps,
                    shots,
                    ways,
                    device,
                    iter,
                )

                kld = kld.sum()
                kld = kld * kl_coeff
                total_l = pseudo_label_error + eval_error_val + refine_error + kld
                classifier_gen_error_tot += classifer_gen_error.item()

                total_l.backward()

                meta_valid_error += (eval_error_val.item()) * beta
                meta_valid_accuracy += eval_accuracy_val.item()

                pseudo_error_val_tot += (pseudo_label_error.item()) * alpha
                refine_error_val_tot += refine_error.item()
                kl_div_val_tot += kld.item()

                if counter == (batch_size - 1):
                    break

            log_string(
                "\t Iteration %d: valid error %.2f, valid accuracy %.2f Pseudo error %.2f Refine error %.2f KL div %.2f classifier_gen_error %.2f"
                % (
                    iter,
                    meta_valid_error / batch_size,
                    meta_valid_accuracy / batch_size,
                    pseudo_error_val_tot / batch_size,
                    refine_error_val_tot / batch_size,
                    kl_div_val_tot / batch_size,
                    classifier_gen_error_tot / batch_size,
                )
            )

            for p in maml.parameters():
                p.grad.data.mul_(1.0 / (batch_size))
            optimizer.step()
            optimizer.zero_grad()

            for counter, (inputs, targets, img_name2) in enumerate(taskset_val):
                learner = maml.clone()

                data, labels = inputs.to(device), targets.to(device)

                eval_error_theta, eval_accuracy_tr = fast_adapt(
                    data, labels, learner, loss, adaptation_steps, shots, ways, device
                )
                eval_error_theta.backward()

                if counter == (batch_size - 1):
                    break

            for p in maml.module.features.parameters():
                p.grad.data.mul_(1.0 / batch_size)
            for p in maml.module.fc.parameters():
                p.grad.data.mul_(1.0 / batch_size)

            optimizer.step()
            optimizer.zero_grad()

        del taskset, taskset_val


best_real_val_acc = 0


def val_meta_model(maml, optimizer, iter, checkpoint_dir):
    global best_real_val_acc

    test_loss = 0
    correct = 0
    total = 0
    ac_correct = [0, 0, 0]

    with torch.no_grad():
        for i in range(4):
            all_dataset.reset("val", i, transform=transform_test)
            valloader = torch.utils.data.DataLoader(
                all_dataset, batch_size=test_batch, shuffle=False, num_workers=0
            )
            num_preds = 1
            for batch_idx, (inputs, targets, img_name1) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                y, _ = maml(inputs)

                cls_loss = criterion(y, targets)
                loss = cls_loss
                test_loss += loss.item()
                _, predicted = y.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        real_val_acc = 100.0 * correct / total
        real_val_loss = test_loss / (batch_idx + 1)

        log_string("\t Real Val Loss %f, Acc: %f" % (real_val_loss, real_val_acc))


decay_ite = [0.6 * max_ite]


train_meta(args)
