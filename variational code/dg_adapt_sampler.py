from __future__ import print_function
from xmlrpc.client import Boolean

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

cpu_workers = psutil.cpu_count()

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


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="learning rate")

parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)

parser.add_argument("--log_dir", default="log1", help="Log dir [default: log]")
parser.add_argument("--dataset", default="PACS", help="datasets")
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="Batch Size during training [default: 32]",
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
    "--test_domain", default="art_painting", help="GPU to use [default: GPU 0]"
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
    default="./new_models/sampler/tt",
    type=str,
    help=" place to save the best model obtained during training",
)

parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    help=" resume from checkpoint",
)
parser.add_argument(
    "--lr_adam_maml", default=0.0001, type=float, help="learning rate Maml"
)
parser.add_argument(
    "--lr_adam", default=0.0001, type=float, help="learning rate Adam Optimizer"
)

parser.add_argument(
    "--entropy_threshold_perc", default=0.7, type=float, help="entropy_threshold_perc"
)
parser.add_argument("--num_iterations", default=250, type=float, help="num_iterations")

parser.add_argument(
    "--variational_refinement", default=True, type=bool, help="variational_refinement"
)

parser.add_argument(
    "--update_pseudo_label_times", default=1, type=int, help="update_pseudo_label_times"
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


data_aug = args.data_aug
data_aug = bool(data_aug)
model_saving_dir = args.model_saving_dir
resume_from_checkpoint = args.resume_from_checkpoint

MODEL_DIR = "./models"
LOG_DIR = "./logs"
WEIGHT_DECAY = 5e-4

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


log_string("Saving models to ", MODEL_DIR)

log_string("==> Writing text file and stdout pushing file output to ")
log_string(text_file)
log_string(text_file2)


tr_writer = SummaryWriter(LOG_DIR)
val_writer = SummaryWriter(os.path.join(LOG_DIR, "validation"))
te_writer = SummaryWriter(os.path.join(LOG_DIR, "test"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-2)
            if m.bias is not None:
                init.constant(m.bias, 0)


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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


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

pc = get_parameter_number(net)
log_string(
    "Total: %.4fM, Trainable: %.4fM"
    % (pc["Total"] / float(1e6), pc["Trainable"] / float(1e6))
)


def fast_adapt_test_time_vi(
    data,
    learner,
    loss,
    adaptation_steps,
    shots,
    ways,
    device,
    iter,
    entropy_threshold_perc,
    update_pseudo_label_times,
):
    labels, _ = learner(data)

    data_np = data.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    if 1 == 1:
        (
            adaptation_data,
            evaluation_data,
            adaptation_labels,
            evaluation_labels,
        ) = train_test_split(data_np, labels_np, test_size=0.5)
        adaptation_data, adaptation_labels = torch.from_numpy(adaptation_data).to(
            device
        ), torch.from_numpy(adaptation_labels).to(device)
        evaluation_data, evaluation_labels = torch.from_numpy(evaluation_data).to(
            device
        ), torch.from_numpy(evaluation_labels).to(device)

        adaptation_steps = 1
        for step in range(adaptation_steps):
            for i in range(update_pseudo_label_times):
                _, features = learner(adaptation_data)
                qDz = []
                for cate in range(7):
                    if cate in adaptation_labels.unique():
                        qDz.append(
                            features[adaptation_labels == cate].mean(0, keepdim=True)
                        )
                    else:
                        qDz.append(features.mean(0, keepdim=True))
                qDz = torch.cat(qDz, 0)
                qw_mu, qw_sigma = learner.module.classifier(qDz)

                y = torch.mm(features, qw_mu.permute(1, 0).contiguous().view(512, 7))

                y = y.view(len(adaptation_labels), 7)

                y = y.unsqueeze(1)

                y = y.repeat(1, 5, 1)

                preds, _ = learner(adaptation_data)

                preds = preds.unsqueeze(2)

                preds = preds.repeat(1, 1, 5)

                _, updated_pseudo_label = torch.max(y, 1)

                unifo = torch.rand(y.size())
                unifo = unifo.to(device)
                samples = torch.argmax(y - torch.log(-torch.log(unifo)), dim=2)

                samples = samples.to(device)

                adaptation_error = loss(preds, samples)
                learner.adapt(adaptation_error)
        predictions, _ = learner(evaluation_data)

        _, features = learner(evaluation_data)
        qDz = []
        for cate in range(7):
            if cate in evaluation_labels.unique():
                qDz.append(features[evaluation_labels == cate].mean(0, keepdim=True))
            else:
                qDz.append(features.mean(0, keepdim=True))
        qDz = torch.cat(qDz, 0)
        qw_mu, qw_sigma = learner.module.classifier(qDz)

        y = torch.mm(features, qw_mu.permute(1, 0).contiguous().view(512, 7))
        y = y.view(len(evaluation_labels), 7)

        _, updated_pseudo_label = torch.max(y, 1)

        pseudo_label_error = loss(predictions, updated_pseudo_label)

        evaluation_accuracy = 0.5
        ent_num_samples = len(labels)

        return pseudo_label_error, evaluation_accuracy, ent_num_samples


def fast_adapt_test_time_vi_normal_refinement(
    data,
    learner,
    loss,
    adaptation_steps,
    shots,
    ways,
    device,
    iter,
    entropy_threshold_perc,
):
    labels, _ = learner(data)

    ent_num_samples = len(labels)

    if len(data) != 0:
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
            preds, _ = learner(adaptation_data)
            _, pseudo_label = torch.max(preds, 1)
            adaptation_error = loss(preds, pseudo_label)
            learner.adapt(adaptation_error)
        predictions, _ = learner(evaluation_data)

        _, pseudo_label = torch.max(predictions, 1)
        alpha = alpha_weight_for_pseudo_label_loss(iter)
        alpha = alpha + 0.5
        pseudo_label_error = alpha * (loss(predictions, pseudo_label))

        evaluation_accuracy = 0.5

        return pseudo_label_error, evaluation_accuracy, ent_num_samples


def eval_test_data(data, labels, learner, loss, adaptation_steps, shots, ways, device):
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
        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)

        return evaluation_error, evaluation_accuracy
    except:
        log_string("Error in fast_adapt, train test split")
        val = 0
        return (
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
            torch.tensor(val).to(device),
        )


def train_meta(args):
    python_file_name = os.path.basename(__file__)

    current_directory = os.getcwd()
    python_file_name = os.path.join(current_directory, python_file_name)

    log_string("Uploaded file: %s" % python_file_name)

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

    batch_size = 8

    adaptation_steps = 1
    num_iterations = args.num_iterations
    shots = 5
    ways = 7
    lr_adam_maml = args.lr_adam_maml
    lr_adam = args.lr_adam
    entropy_threshold_perc = args.entropy_threshold_perc
    update_pseudo_label_times = args.update_pseudo_label_times

    fast_lr = 0.5

    maml = l2l.algorithms.MAML(
        net, lr=lr_adam_maml, first_order=False, allow_nograd=True
    )
    maml.to(device)
    print(maml)

    optimizer = torch.optim.Adam(
        maml.parameters(),
        lr=lr_adam,
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
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            maml.load_state_dict(checkpoint["net"])

            best_valid_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]
            log_string("Loaded checkpoint from epoch {}".format(start_epoch))
            log_string("\n \n ===> Loaded model from checkpoint <==========")
        else:
            log_string("No checkpoint found at {}".format(args.resume_from_checkpoint))
            exit()
    maml.to(device)

    all_dataset.reset("ttt", 0, transform=transform_test)

    test_dataset = l2l.data.MetaDataset(all_dataset)
    transforms_test = [
        l2l.data.transforms.NWays(test_dataset, ways),
        l2l.data.transforms.KShots(test_dataset, 2 * shots),
        l2l.data.transforms.LoadData(test_dataset),
    ]
    test_taskset = l2l.data.TaskDataset(test_dataset, transforms_test, num_tasks=100)

    count_epoch_acc_best = 0

    for iter in range(num_iterations):
        t0 = time.time()

        if int(math.remainder(iter, 1)) == 0:
            meta_test_error = 0
            meta_test_accuracy = 0
            ent_num_samples_tot = 0
            for counter, (inputs, targets, img_name2) in enumerate(test_taskset):
                learner = maml.clone()

                data, labels = inputs.to(device), targets.to(device)

                if (
                    args.variational_refinement == True
                    and update_pseudo_label_times != 0
                ):
                    (
                        eval_error_test,
                        eval_accuracy_test,
                        ent_num_samples,
                    ) = fast_adapt_test_time_vi(
                        data,
                        learner,
                        loss,
                        adaptation_steps,
                        shots,
                        ways,
                        device,
                        iter,
                        entropy_threshold_perc,
                        update_pseudo_label_times,
                    )
                elif update_pseudo_label_times == 0:
                    (
                        eval_error_test,
                        eval_accuracy_test,
                        ent_num_samples,
                    ) = fast_adapt_test_time_vi_normal_refinement(
                        data,
                        learner,
                        loss,
                        adaptation_steps,
                        shots,
                        ways,
                        device,
                        iter,
                        entropy_threshold_perc,
                    )
                eval_error_test = 0.0001 * eval_error_test
                eval_error_test.backward()
                meta_test_error += eval_error_test.item()
                ent_num_samples_tot += ent_num_samples

                if counter == (batch_size - 1):
                    break
            log_string("\t Iteration")

            for p in maml.module.features.parameters():
                p.grad.data.mul_(1.0 / batch_size)
            for p in maml.module.fc.parameters():
                p.grad.data.mul_(1.0 / batch_size)
            optimizer.step()
            optimizer.zero_grad()
        t1 = time.time()


train_meta(args)
