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

:exclamation: This repo only contains configs and experiments on small or medium scale datasets such as CIFAR-10/100 and Tiny-ImageNet. Full re-implementation on ImageNet-1k would be extremely expensive.

## Requirements
In addition to PyTorch environments, please install:
```sh
conda install pyyaml
pip install ema-pytorch tensorboard
```

## Issues, different implementations, and TODOs
### Issues
- Current version of this implementation fails to achieve SOTA-level classification performances on the small or medium scale datasets. Maybe it's because these images are not diverse enough.
- The image reconstruction is not satisfactory, because augmentations such as RandomResizedCrop and RandAugment are also applied to the target (decoder) inputs. Removing those augmentations lead to improved reconstruction but degraded classification performances.

### Different implementations
- Weight initializations (Xavier for Resnet, truncated normal for Unet) are not adopted, since I don't find them helpful on these datasets.
- $\sqrt2$ rescaling of Unet blocks is not adopted, since I don't find them helpful on these datasets.
- The overall architecture of the Unet decoder follows standard DDPM implementations, except for the modulations.

### TODOs
- Layer Modulation & Masking

## Main results
|    Model   |    Dataset    |  Resolution | Epochs | #Params | K-NN acc | Linear probe acc |
|------------|---------------|-------------|--------|---------|----------|------------------|
| Res18+DDPM |   CIFAR-10    |    32x32    |   800  |  11+40  |   80.4   |      80.0        |
| Res18+DDPM |   CIFAR-100   |    32x32    |   800  |  11+40  |   51.4   |      54.9        |
| Res18+DDPM | Tiny-ImageNet |    64x64    |   800  |  11+40  |   34.8   |      38.2        |
<p align="center">
  <img src="https://github.com/FutureXiang/soda/assets/33350017/2caaac7e-4e4e-4d6e-952f-e0ed5e1e55c9" width="800">
</p>

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

During training, the SODA encoder is evaluated by a K-NN classifier or linear probing every 100 epoch.
Typically, we should not evaluate checkpoints on the validation set like this, but here we just want to observe and get a better understanding of SODA.

To evaluate the final checkpoint after training, run:
```sh
python 
  test.py   --config config/cifar10.yaml  --use_amp
  test.py   --config config/cifar100.yaml --use_amp
  test.py   --config config/tiny.yaml     --use_amp
```
