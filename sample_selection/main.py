# -*- coding:utf-8 -*-
import os
import csv
import argparse
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F

from data.datasets import input_dataset_clip
from data.transform import *
from models.nine_layer_cnn import CNN
from models.resnet import get_resnet_type as ResNet
from models.clip import clip as CLIP
from models.dino import DinoV2
from models.simclr.ssl import SSLEval as SimCLR
from loss import loss_cross_entropy, loss_cores, f_beta, adjust_learning_rate

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from training.augment import AugmentPipe

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--lr_plan', type=str, help='base, cyclic', default='cyclic')
parser.add_argument('--loss', type=str, help='ce, cores', default='cores')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric,instance]', default='pairflip')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--data_dir', type=str, help='dir to data files', default='./data')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar10')
# config for SimCLR
parser.add_argument('--pretrain_on', type=str, help='cifar or imagenet', default='imagenet')
parser.add_argument('--encoder_ckpt', type=str, default='', help='Path to the encoder checkpoint')
parser.add_argument('--model_type', type=str, default='resnet50', help='Architecture used by each method')
parser.add_argument('--finetune', default=False, type=bool, help='Finetunes the encoder if True')
# ----------------------------------------------------------
parser.add_argument('--num_classes', type=int, help='Number of classes', default=10)
parser.add_argument('--augment', type=float, help='Augment probability', default=0.12)
parser.add_argument('--model', type=str, help='cnn,resnet,simclr,clip,dino', default='cnn')

parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--save_ckpt', action='store_true', help='save checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')


def get_noise_pred(loss_div, args, epoch=-1, alpha=0.):
    # Get noise prediction
    print('DEBUG, loss_div', loss_div.shape)
    llast = loss_div[:, epoch]
    idx_last = np.where(llast > alpha)[0]
    print('last idx:', idx_last.shape)
    return idx_last


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision_recall_f1(idx_noisy_each_class, idx_clean_each_class, noise_or_not):
    idx_noisy_all_class = np.hstack(idx_noisy_each_class)
    precision = torch.tensor(0.)
    # precision_class_noise = [np.inf for _ in range(len(idx_noisy_each_class))]
    true_positive = torch.tensor(1.)
    if len(idx_noisy_all_class) != 0:
        noise_all_class = torch.tensor(noise_or_not)[idx_noisy_all_class]
        true_positive = noise_all_class.sum().clamp(1)
        total_positive_pred = len(noise_all_class)
        precision = true_positive / total_positive_pred

    idx_clean_all_class = np.hstack(idx_clean_each_class)
    clean_all_class = torch.tensor(noise_or_not)[idx_clean_all_class]
    false_negative = clean_all_class.sum()
    recall = true_positive / (true_positive + false_negative)

    return {"precision": round(precision.item(), ndigits=4),
            "recall": round(recall.item(), ndigits=4),
            "f1": round((2 * precision * recall / (precision + recall)).item(), ndigits=4),
            }


# Train the Model
def train(
        epoch,
        num_classes,
        train_loader,
        model,
        optimizer,
        loss_all,
        loss_div_all,
        loss_type,
        augment=None,
        noise_prior=None,
):
    train_total = 0
    train_correct = 0
    print(f"current beta is {f_beta(epoch)}")
    v_list = np.zeros(num_training_samples)
    idx_noisy_each_class = [[] for i in range(num_classes)]
    idx_clean_each_class = [[] for i in range(num_classes)]
    if not isinstance(noise_prior, torch.Tensor):
        noise_prior = torch.tensor(noise_prior.astype("float32")).cuda().unsqueeze(0)
    for i, (images, labels, indexes) in tqdm(enumerate(train_loader)):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        class_list = range(num_classes)
        images = Variable(images).cuda()
        if augment:
            images, _ = augment(images)
        labels = Variable(labels).long().cuda()

        # Forward + Backward + Optimize
        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec
        if loss_type == "ce":
            loss = loss_cross_entropy(
                epoch,
                logits,
                labels,
                class_list,
                ind,
                noise_or_not,
                loss_all,
                loss_div_all,
            )
        elif loss_type == "cores":
            loss, loss_v = loss_cores(
                epoch,
                logits,
                labels,
                class_list,
                ind,
                noise_or_not,
                loss_all,
                loss_div_all,
                noise_prior=noise_prior,
            )
            v_list[ind] = loss_v
            for i in range(batch_size):
                if loss_v[i] == 0:
                    idx_noisy_each_class[labels[i]].append(ind[i])
                else:
                    idx_clean_each_class[labels[i]].append(ind[i])
        else:
            print("loss type not supported")
            raise SystemExit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % args.print_freq == 0:
            print(
                "Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f"
                % (
                    epoch + 1,
                    args.n_epoch,
                    i + 1,
                    len(train_dataset) // batch_size,
                    prec,
                    loss.data,
                )
            )

    class_size_noisy = [len(idx_noisy_each_class[i]) for i in range(num_classes)]
    noise_prior_delta = np.array(class_size_noisy)
    print(noise_prior_delta)

    train_acc = float(train_correct) / float(train_total)
    return train_acc, noise_prior_delta, idx_noisy_each_class, idx_clean_each_class, v_list 


# Evaluate the Model
def evaluate(test_loader, model, save=False, epoch=0, best_acc_=0, args=None, save_data=None):
    model.eval()  # Change model to 'eval' mode.
    print("previous_best", best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)

    if save:
        if acc > best_acc_:
            state = {
                "epoch": epoch,
                "acc": acc,
            }
            torch.save(
                state,
                os.path.join(
                    save_dir,
                    args.loss + args.noise_type + str(args.noise_rate) + "best.pth.tar",
                ),
            )
            if save_data is not None:
                np.savez(os.path.join(save_dir, args.loss + args.noise_type + str(args.noise_rate) + "best_discriminate_data"), *save_data)
            # np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_div_all_best.npy',loss_div_all)
            # np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_all_best.npy',loss_all)
            best_acc_ = acc
        if epoch == args.n_epoch - 1:
            state = {
                "epoch": epoch,
                "acc": acc,
            }
            torch.save(
                state,
                os.path.join(
                    save_dir,
                    args.loss + args.noise_type + str(args.noise_rate) + "last.pth.tar",
                ),
            )
            np.savez(os.path.join(save_dir, args.loss + args.noise_type + str(args.noise_rate) + "last_discriminate_data"), *save_data)
            # np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_div_all_last.npy',loss_div_all)
            # np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_all_best.npy',loss_all)
    return acc, best_acc_


#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 64
learning_rate = args.lr

# load model
model_input_size = -1
print("building model...")
if args.model == "cnn":
    model = CNN(input_channel=3, n_outputs=args.num_classes)
elif args.model == "resnet":
    model = ResNet(args.model_type, args.num_classes)
elif args.model == "simclr":
    model = SimCLR(args, device="cpu", n_classes=args.num_classes)
elif args.model == "clip":
    model = CLIP.load(args.model_type, "cpu", jit=False,
                      logits=True, num_classes=args.num_classes
                      )  # RN50, RN101, RN50x4, ViT-B/32
    model_input_size = model.visual.input_resolution
elif args.model == "dino":
    model = DinoV2(args.model_type, num_classes=args.num_classes)
else:
    assert "Not any model type in [CNN, ResNet, Sim-CLR, CLIP, DinoV2]"
print("building model done")
# print(model)
# alpha_plan = [0.1] * 50 + [0.01] * 50
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("building optimizer done")

# set transforms
augment_pipeline = None
if args.augment != 0:
    print("Strong augmentations are used!!")
    augment_pipeline = AugmentPipe(p=args.augment, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1,
                                   translate_frac=1)  # disable rotate_int, translate_int, 5 color trans
preprocess_rand = get_data_and_model_transform(args.dataset, args.model, model_input_size)

# load dataset
train_dataset, test_dataset, num_classes, num_training_samples = input_dataset_clip(
    args.data_dir, args.dataset, args.noise_type, args.noise_rate,
    transform=transforms.Compose(preprocess_rand),
)
train_data = train_dataset.train_data
train_labels = train_dataset.train_noisy_labels
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print("train_labels:", len(train_dataset.train_labels), train_dataset.train_labels[:10])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, num_workers=args.num_workers, shuffle=False,
)

# Creat loss and loss_div for each sample at each epoch
loss_all = np.zeros((num_training_samples, args.n_epoch))
loss_div_all = np.zeros((num_training_samples, args.n_epoch))
### save result and model checkpoint #######
save_dir = args.result_dir + "/" + args.dataset + "/" + args.model 
if not os.path.exists(save_dir):
    os.system("mkdir -p %s" % save_dir)
# alpha_plan = []
# for ii in range(args.n_epoch):
#    alpha_plan.append(learning_rate*pow(0.95,ii))
txtfile = save_dir + "/" + args.loss + args.noise_type + str(args.noise_rate) + ".txt"
logfile = save_dir + "/" + args.loss + args.noise_type + str(args.noise_rate) + ".csv"
if os.path.exists(txtfile):
    os.system("rm %s" % txtfile)
with open(txtfile, "a") as myfile:
    myfile.write("epoch: train_acc test_acc \n")
fieldnames = ["epoch", "beta", "train acc", "test acc",
              "precision", "recall", "f1"]
if os.path.exists(logfile):
    os.system("rm %s" % logfile)
with open(logfile, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

epoch = 0
train_acc = 0
best_acc_ = 0.0
# print(best_acc_)
# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    print(f"Current epoch: {epoch}")
    # train models
    adjust_learning_rate(optimizer, epoch)
    model.train()
    train_acc, noise_prior_delta, idx_noisy_each_class, idx_clean_each_class, real_or_not_predict = train(
        epoch,
        num_classes,
        train_loader,
        model,
        optimizer,
        loss_all,
        loss_div_all,
        args.loss,
        augment=augment_pipeline,
        noise_prior=noise_prior_cur,
    )
    noise_prior_cur = noise_prior * num_training_samples - noise_prior_delta
    noise_prior_cur = noise_prior_cur / sum(noise_prior_cur)
    # evaluate models
    test_acc, best_acc_ = evaluate(
        test_loader=test_loader,
        save=args.save_ckpt,
        model=model,
        epoch=epoch,
        best_acc_=best_acc_,
        args=args,
        save_data=[train_dataset.train_data, train_labels, real_or_not_predict]
    )
    metrics = precision_recall_f1(
        idx_noisy_each_class,
        idx_clean_each_class,
        noise_or_not
    )
    # save results
    # det_by_loss  = det_acc(save_dir,best_ratio,args,loss_all,noise_or_not,epoch,sum_epoch = False)
    # det_by_loss_div = det_acc(save_dir,best_ratio,args,loss_div_all,noise_or_not,epoch,sum_epoch = False)
    with open(logfile, 'a') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=fieldnames)
        writer.writerows(
            [dict(
                zip(fieldnames,
                    [epoch, f_beta(epoch), train_acc, test_acc,
                     metrics["precision"],  metrics["recall"], metrics["f1"]
                     ]
                    )
            )
            ]
        )
    print("train acc on train images is ", train_acc)
    print("test acc on test images is ", test_acc)
    print("noise detection on train images is ")
    print("precision: ", metrics["precision"], "recall: ", metrics["recall"], "f1: ", metrics["f1"])
    print("*" * 50)
    # print('precision of labels by loss is', det_by_loss)
    # print('precision of labels by loss div is', det_by_loss_div)
    with open(txtfile, "a") as myfile:
        myfile.write(
            str(int(epoch)) + ": " + str(train_acc) + " " + str(test_acc) + "\n"
        )
    np.save(
        save_dir
        + "/"
        + args.loss
        + args.noise_type
        + str(args.noise_rate)
        + "loss_div_all.npy",
        loss_div_all,
    )
    np.save(
        save_dir
        + "/"
        + args.loss
        + args.noise_type
        + str(args.noise_rate)
        + "noise_or_not.npy",
        noise_or_not,
    )   # ok
    np.save(
        save_dir
        + "/"
        + args.loss
        + args.noise_type
        + str(args.noise_rate)
        + "train_noisy_labels.npy",
        train_dataset.train_noisy_labels,
    )   # ok
    if epoch == 15:
        idx_last = get_noise_pred(loss_div_all, args, epoch=epoch)
        np.save(
            save_dir
            + "/"
            + args.loss
            + args.noise_type
            + str(args.noise_rate)
            + "_noise_pred.npy",
            idx_last,
        )
