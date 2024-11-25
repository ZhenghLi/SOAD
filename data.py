import os
import os.path
import cv2
import glob
import numpy as np
import torch
import utils

DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn

@register_dataset("OCTA")
def load_OCTA(data, batch_size=100, num_workers=4, image_size=None, stride=64, n_frames=7, padding=True):
    train_dataset = OCTA(data, patch_size=image_size, stride=stride, n_frames=n_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    valid_dataset = OCTA_val(data, n_frames=n_frames, padding=padding)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, valid_loader

@register_dataset("OCTA_val")
def load_OCTA_val(data, num_workers=4, n_frames=7, padding=True):
    valid_dataset = OCTA_val(data, n_frames=n_frames, padding=padding)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return valid_loader

class OCTA(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, stride=64, n_frames=7):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames
        self.data_cache = []

        self.volume_ids = []
        self.folders = sorted([x for x in glob.glob(os.path.join(data_path, "OCTA/*")) if os.path.isdir(x)])

        for i, folder in enumerate(self.folders):
            self.volume_ids.append(i)
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            if self.size is not None:
                (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
                nH = (int((h - self.size) / self.stride) + 1)
                nW = (int((w - self.size) / self.stride) + 1)
                self.len += (len(files) - n_frames + 1) * nH * nW
                self.nHs.append(nH)
                self.nWs.append(nW)
            else:
                self.len += len(files)

            volume = []
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                volume.append(img)
            volume = np.stack(volume, axis=0)
            self.data_cache.append(volume)

            self.bounds.append(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                volume_id = self.volume_ids[i-1]
                index -= self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                break

        files = sorted(glob.glob(os.path.join(folder, "*.tif")))

        nh = (patch // nW) * self.stride
        nw = (patch % nW) * self.stride

        volume  = self.data_cache[volume_id][index:index+self.n_frames, nh:(nh+self.size), nw:(nw+self.size)]

        (n, h, w) = volume.shape
        volume = np.reshape(volume, (1, n, h, w)).astype('float')
        volume = volume / 255
        Volume = torch.from_numpy(volume).type(torch.FloatTensor)

        target = Volume[:, x, :, :].clone()

        _, fname = os.path.split(files[index+x])
        target_mask_id = int(fname.split('.')[0].split('_')[1])  # can be just index+x if all B-scans are included
        # assert target_mask_id == index + x
        target_mask_volume = fname.split('_')[0]

        bma_path = os.path.join(os.path.join(self.data_path, "OCTA"), target_mask_volume + '_BMA_index.txt')
        bmas = None
        assert os.path.exists(bma_path)
        if os.path.exists(bma_path):
            bmas = np.loadtxt(bma_path).reshape(-1).astype('int')

        if bmas is None or target_mask_id not in bmas:
            flag = torch.ones(1, dtype=torch.bool)
        else:
            flag = torch.zeros(1, dtype=torch.bool)

        return Volume, target, flag

class OCTA_val(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, n_frames=7, padding=True):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames
        self.padding = padding

        self.folders = sorted([x for x in glob.glob(os.path.join(data_path, "OCTA/*")) if os.path.isdir(x)])
        if padding:
            for folder in self.folders:
                files = sorted(glob.glob(os.path.join(folder, "*.tif")))
                self.len += len(files)
                self.bounds.append(self.len)
        else:
            for folder in self.folders:
                files = sorted(glob.glob(os.path.join(folder, "*.tif")))
                self.len += (len(files) - n_frames + 1)
                self.bounds.append(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                index -= self.bounds[i-1]
                break

        files = sorted(glob.glob(os.path.join(folder, "*.tif")))

        Img = []
        if self.padding:
            for i in range(self.n_frames):
                j = i - (self.n_frames // 2)
                img_id = index + j
                img_id = min(max(0, img_id), len(files) - 1)
                img = cv2.imread(files[img_id], cv2.IMREAD_GRAYSCALE).astype('float')
                img = img[None, :, :] / 255
                img = torch.from_numpy(img).float()
                Img.append(img)
            _, fname = os.path.split(files[index])
            target_name = fname
        else:
            for i in range(self.n_frames):
                img = cv2.imread(files[index + i], cv2.IMREAD_GRAYSCALE).astype('float')
                img = img[None, :, :] / 255
                img = torch.from_numpy(img).float()
                Img.append(img)
            _, fname = os.path.split(files[index+x])
            # assert int(fname.split('_')[1]) == index + x
            target_name = fname
        
        Img = torch.stack(Img, dim=1) # C,T,H,W

        return Img, target_name

if __name__ == "__main__":
    train_loader, valid_loader = build_dataset('OCTA', 'data', batch_size=8, image_size=128, stride=64, n_frames=7)
    for epoch in range(0, 10):
        train_bar = utils.ProgressBar(train_loader, epoch)

        for batch_id, (inputs, targets, flags) in enumerate(train_bar):
            pass

        valid_bar = utils.ProgressBar(valid_loader)
        for sample_id, (sample, target_name) in enumerate(valid_bar):
            pass