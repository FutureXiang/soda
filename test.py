import argparse
import os
from functools import partial

import torch
import yaml
from metric import KNN, LinearProbe
from ema_pytorch import EMA

from model.SODA import SODA
from model.encoder import Network
from model.decoder import UNet_decoder
from utils import Config, init_seeds
from datasets import get_dataset


def get_model(opt, load_epoch):
    soda = SODA(encoder=Network(**opt.encoder),
                decoder=UNet_decoder(**opt.decoder),
                **opt.diffusion,
                device=device)
    soda.to(device)
    target = os.path.join(opt.save_dir, "ckpts", f"model_{load_epoch}.pth")
    print("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    ema = EMA(soda, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'])
    model = ema.ema_model
    model.eval()
    return model


def test(opt):
    yaml_path = opt.config
    use_amp = opt.use_amp
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)

    num_classes, _, down_train, down_test = get_dataset(name=opt.dataset, root="./data")
    knn = KNN(down_train, down_test)
    lp = LinearProbe(down_train, down_test, num_classes)

    model = get_model(opt, opt.n_epoch - 1)
    feat_func = partial(model.encode, norm=True, use_amp=use_amp)
    knn_acc = knn.evaluate(feat_func)
    lp_acc = lp.evaluate(feat_func)
    print(f"knn:{knn_acc * 100:.1f}, linear:{lp_acc * 100:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print(opt)

    init_seeds(no=0)
    device = "cuda:%d" % 0
    test(opt)
