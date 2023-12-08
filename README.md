# SODA: Bottleneck Diffusion Models for Representation Learning
<p align="center">
  <img src="https://github.com/FutureXiang/soda/assets/33350017/7bfd19a8-950b-44f1-8d36-d3c7e0866321" width="640">
</p>

This is a multi-gpu PyTorch implementation of the paper [SODA: Bottleneck Diffusion Models for Representation Learning](https://arxiv.org/abs/2311.17901):
```bibtex
@article{hudson2023soda,
  title={SODA: Bottleneck Diffusion Models for Representation Learning},
  author={Hudson, Drew A and Zoran, Daniel and Malinowski, Mateusz and Lampinen, Andrew K and Jaegle, Andrew and McClelland, James L and Matthey, Loic and Hill, Felix and Lerchner, Alexander},
  journal={arXiv preprint arXiv:2311.17901},
  year={2023}
}
```
:exclamation: Note that this implementation only cares about the *linear-probe classification* performance, and somewhat ignores other generative downstream tasks. However, this could be a good start for further development. Please check out [this DDAE repo](https://github.com/FutureXiang/ddae), which is the "unconditional" baseline in the SODA paper, if you are also interested in diffusion-based classification.

:exclamation: This repo only contains configs and experiments on small-size datasets such as CIFAR-10/100 and Tiny-ImageNet. Full re-implementation on ImageNet-1k would be extremely expensive.

## Requirements
In addition to PyTorch environments, please install:
```sh
conda install pyyaml
pip install ema-pytorch tensorboard
```

## Main results
(To be updated).

## Usage
Use 4 GPUs to train `SODA = resnet18 + DDPM` on CIFAR-10/100 classification:
```sh
python -m torch.distributed.launch --nproc_per_node=4
  train.py  --config config/cifar10.yaml  --use_amp
  train.py  --config config/cifar100.yaml --use_amp
```
Use more GPUs to train `SODA = resnet18 + DDPM` on Tiny-ImageNet classification:
```sh
python -m torch.distributed.launch --nproc_per_node=8
  train.py  --config config/tiny.yaml     --use_amp
```

During training, the SODA encoder is evaluated by a K-NN classifier or Linear Probing every 100 epoch.
Typically, we should not evaluate checkpoints on the validation set like this, but here we just want to observe and get a better understanding of SODA.
