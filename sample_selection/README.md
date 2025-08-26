# Learning with Instance-Dependent Label Noise: A Sample Sieve Approach
This code is a PyTorch utilization of the paper "[Learning with Instance-Dependent Label Noise: A Sample Sieve Approach](https://arxiv.org/abs/2010.02347)" accepted by ICLR2021.

The code is run on one A100.

## Steps on Runing CORES on CIFAR 10
### Step 1: 

Download the datset from **http://www.cs.toronto.edu/~kriz/cifar.html** Put the dataset on **data/**

Install theconf by **pip install git+https://github.com/wbaek/theconf.git**


### Step 2:

Run CORES (Phase 1: Sample Sieve) on CIFAR-10 with symmetric 0.5 noise:

```
CUDA_VISIBLE_DEVICES=1 python main.py --loss cores --dataset cifar10 --data_dir /cm/shared/datnc13/noisy_data/CIFAR10_50.0_sym.pk  --model clip \
--noise_type sym --noise_rate 0.5
```
### Step 3:
Run CORES (Phase 2: Consistency Training) on the CIFAR-10 with symmetric 0.5 noise:

```
cd phase2
CUDA_VISIBLE_DEVICES=0,1 python phase2.py -c confs/resnet34_sym_0.5.yaml --unsupervised
```
**Both Phase 1 and Phase 2 do not need pre-trained model.**


## Citation

If you find this code useful, please cite the following paper:

```
@article{cheng2020learning,
  title={Learning with Instance-Dependent Label Noise: A Sample Sieve Approach},
  author={Cheng, Hao and Zhu, Zhaowei and Li, Xingyu and Gong, Yifei and Sun, Xing and Liu, Yang},
  journal={arXiv preprint arXiv:2010.02347},
  year={2020}
}
```


## References

The code of Phase 2 is based on **https://github.com/ildoonet/unsupervised-data-augmentation**






