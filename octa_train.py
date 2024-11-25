import argparse
import os.path

import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from vnet import VNetProjnl

import data, utils

def weight_mse(input, target, alpha=4.5, beta=0.5, gamma=2, epsilon=0.05, reduction='sum'):
    mse = (input - target) ** 2
    input_data = input.detach()
    input_data[input_data < 0] = 0
    input_data[input_data > 1] = 1
    target_data = target.detach()
    weights = alpha*torch.pow(input_data, 1.0/gamma) + beta*torch.pow(target_data, 1.0/gamma) + epsilon
    weighted_mse = weights*mse
    if reduction == 'sum':
        return torch.sum(weighted_mse)
    elif reduction == 'mean':
        return torch.mean(weighted_mse)
    else:
        return weighted_mse

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    epsilon = args.epsilon
    model = VNetProjnl(1, 1, args.n_frames)
    model = model.to(device)
    mid = args.n_frames // 2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12], gamma=0.1)

    global_step = -1
    start_epoch = 0

    train_loader, valid_loader = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size,
                                                    image_size=args.image_size, stride=args.stride,
                                                    n_frames=args.n_frames, padding=args.padding)
    
    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss"])}

    for epoch in range(start_epoch, args.num_epochs):
        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, (inputs, targets, flags) in enumerate(train_bar):
            model.train()

            flags = flags.squeeze(1)

            global_step += 1

            inputs[:, :, mid, :, :] = 0
            inputs = inputs.to(device)
            targets = targets.to(device)

            out = model(inputs)

            if torch.sum(flags) != 0:
                loss = weight_mse(out[flags], targets[flags], alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon, reduction='sum') / torch.sum(flags).item()
            else:
                model.zero_grad()
                continue

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.sum(flags) != 0:
                train_meters["train_loss"].update(loss.item())
            
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
            
        torch.save(model.state_dict(), 'ckpt.pth')

        scheduler.step()

        if (epoch + 1) % args.valid_interval == 0:
            model.eval()

            if not os.path.exists('val_out'):
                os.mkdir('val_out')

            valid_bar = utils.ProgressBar(valid_loader)
            for sample_id, (sample, target_name) in enumerate(valid_bar):
                with torch.no_grad():
                    sample[:, :, mid, :, :] = 0
                    noisy_inputs = sample.to(device)
                    out = model(noisy_inputs)
                    img = out.cpu().squeeze().numpy()
                    img[img < 0] = 0
                    img[img > 1] = 1
                    img = img * 255
                    img = img.astype('uint8')
                    volume_id = target_name[0].split('_')[0] + '_val'
                    save_dir = os.path.join('val_out', volume_id)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    cv2.imwrite(os.path.join(save_dir, target_name[0]), img)

        if epoch + 1 == args.num_epochs:
            torch.save(model.state_dict(), 'ckpt_' + str(epoch).zfill(2) + '.pth')

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="OCTA", help="train dataset name")
    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")
    parser.add_argument("--image-size", default=128, type=int, help="image size for train")
    parser.add_argument("--n-frames", default=7, type=int, help="number of frames for training")
    parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=15, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    
    # Add loss arguments
    parser.add_argument("--alpha", default=100, type=float, help="alpha")
    parser.add_argument("--beta", default=1, type=float, help="beta")
    parser.add_argument("--gamma", default=3, type=float, help="gamma")
    parser.add_argument("--epsilon", default=0.5, type=float, help="epsilon")

    # Add validation arguments
    parser.add_argument("--padding", action='store_true', help="whether to replicate the boundary B-scans during validation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
