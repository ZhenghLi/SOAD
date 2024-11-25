import argparse
import os.path

import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from vnet import VNetProjnl

import data, utils


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = VNetProjnl(1, 1, args.n_frames)
    model = model.to(device)
    model.load_state_dict(torch.load(args.ckpt))
    mid = args.n_frames // 2
    # print(model)

    valid_loader = data.build_dataset(args.dataset, args.data_path, n_frames=args.n_frames, padding=args.padding)

    model.eval()

    if not os.path.exists('test_out'):
        os.mkdir('test_out')

    valid_bar = utils.ProgressBar(valid_loader)
    for sample_id, (sample, target_name) in enumerate(valid_bar):
        with torch.no_grad():
            sample[:, :, mid, :, :] = 0
            noisy_inputs = sample.to(device)
            out = model(noisy_inputs)
            img = out.cpu().squeeze().numpy()
            img[img < 0] = 0
            img = img * 255
            img = img.astype('uint8')
            volume_id = target_name[0].split('_')[0]
            save_dir = os.path.join('test_out', volume_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, target_name[0]), img)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="OCTA_val", help="train dataset name")
    parser.add_argument("--n-frames", default=7, type=int, help="number of frames for training")
    parser.add_argument("--padding", action='store_true', help="whether to replicate the boundary B-scans during inference")
    parser.add_argument("--ckpt", default='ckpt_14.pth', help="path to checkpoint")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
