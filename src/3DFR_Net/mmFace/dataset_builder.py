import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from numpy.random import permutation
from utils import get_crd_data, by_experiment
from tqdm import tqdm
from glob import glob
from more_itertools import chunked
import numpy as np
import json
import os

def build_dataset(raw_path, subjects=[0], num_frames=250, train_split=0.8, test_split=0.1, video=False, seed=333):
    np.random.seed(seed)
    folder_name = f"3D-{num_frames}" if video else num_frames
    frame_counts = [[], [], []]
    val_split_end = train_split + test_split
    for subject in tqdm(subjects):
        if not os.path.exists(f"data/{folder_name}/train/{subject}.npy"):
            train, val, test = [], [], []
            radar_paths = sorted(glob(rf"{raw_path}\{subject}\*_radar.json"), key=by_experiment)
            for rp in radar_paths:
                with open(rp, 'r') as f:
                    # Extract reshaped ARD for each experiment (Frames x Range (H) x Doppler (W) x Channel)
                    experiment_ard = np.abs(get_crd_data(json.load(f), num_chirps_per_burst=16)[:num_frames]).astype(np.float32)

                    # Frames split for Train/Validation/Test
                    frames = list(range(len(experiment_ard)))    
                    np.random.shuffle(frames)
                    
                    # TODO: FOR VIDEO, NEED TO CHUNK EVENLY THEN SHUFFLE
                    train_idx = frames[:int(len(frames)*train_split)]
                    val_idx = frames[int(len(frames)*train_split):int(len(frames)*(val_split_end))]
                    test_idx = frames[int(len(frames)*(val_split_end)):]

                    train.append(experiment_ard[train_idx])
                    val.append(experiment_ard[val_idx])
                    test.append(experiment_ard[test_idx])
            
            train = np.concatenate(train)
            val = np.concatenate(val)
            test = np.concatenate(test)

            np.save(f"data/{folder_name}/train/{subject}.npy", train)
            np.save(f"data/{folder_name}/validation/{subject}.npy", val)
            np.save(f"data/{folder_name}/test/{subject}.npy", test)
            frame_counts[0].append(str(train.shape[0]))
            frame_counts[1].append(str(val.shape[0]))
            frame_counts[2].append(str(test.shape[0]))
    
    if len(frame_counts[0]) > 0:
        new_line = '\n' if os.path.exists(f"data/{folder_name}/frame_counts_train.txt") else ''
        with open(f"data/{folder_name}/frame_counts_train.txt", 'a') as f:
            f.write(new_line + '\n'.join(frame_counts[0]))
        with open(f"data/{folder_name}/frame_counts_validation.txt", 'a') as g:
            g.write(new_line + '\n'.join(frame_counts[1]))
        with open(f"data/{folder_name}/frame_counts_test.txt", 'a') as h:
            h.write(new_line + '\n'.join(frame_counts[2]))

def normalise(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

class MMFaceDataset(Dataset):
    def __init__(self, data_path, split="train", subjects=[0], num_frames=250, transform=None):
        if len(os.listdir(f"data/{num_frames}/{split}")) != len(subjects):
            build_dataset(data_path, subjects, num_frames)
        
        self.split = split
        self.subjects = subjects
        self.num_frames = num_frames
        self.transform = transform
        with open(f"data/{num_frames}/frame_counts_{split}.txt", 'r') as f:
            self.frame_counts = [int(x) for x in f.read().splitlines()]
    
    def __len__(self):
        return sum(self.frame_counts)
    
    def __getitem__(self, idx):
         # Given an overall index, find the corresponding file index and subject
        _file_idx = lambda self, i, s=0: (s, i) if s >= len(self.frame_counts) or i < self.frame_counts[s] else _file_idx(self, i-self.frame_counts[s], s+1)
        subject, mod_idx = _file_idx(self, idx, 0)
        data = np.load(f"data/{self.num_frames}/{self.split}/{subject}.npy")[mod_idx]
        if self.transform:
            data = self.transform(data)
        
        return data, subject

# NB: SLOW, ONLY USE WHEN DATASET TO BIG TO BE LOADED IN GPU
def load_dataset_DL(data_path, subjects=[0], num_frames=250, batch_size=32, seed=42, transform=ToTensor()):
    train_dataset = MMFaceDataset(data_path, "train", subjects, num_frames, transform)
    val_dataset = MMFaceDataset(data_path, "validation", subjects, num_frames, transform)
    test_dataset = MMFaceDataset(data_path, "test", subjects, num_frames, transform)
    
    np.random.seed(seed)
    # _ Subjects x 15 Scenarios x 250 total frames
    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(permutation(len(train_dataset))))
    val_loader = DataLoader(val_dataset, batch_size, sampler=SubsetRandomSampler(permutation(len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(permutation(len(test_dataset))))


    return train_loader, val_loader, test_loader

def batched_idxs(seed, lengths, batch_size):
    np.random.seed(seed)
    train_idx = list(chunked(np.random.permutation(lengths[0]), batch_size))
    val_idx = list(chunked(np.random.permutation(lengths[1]), batch_size))
    test_idx = list(chunked(np.random.permutation(lengths[2]), batch_size))

    return train_idx, val_idx, test_idx

def load_experiments(folder_name, subject, experiments, split, device):
    subject_data = torch.tensor(np.load(f"data/{folder_name}/{split}/{subject}.npy"))
    exp_batch_size = len(subject_data) // 15
    batched = list(chunked(subject_data, exp_batch_size))
    data = np.vstack([batched[e] for e in experiments])

    return torch.tensor(data, device=device)

def load_dataset(data_path, subjects, experiments=list(range(15)), num_frames=250, batch_size=64, seed=42, device="cuda", video=False, frame_batch_size=16):
    folder_name = f"3D-{num_frames}" if video else num_frames
    if len(os.listdir(f"data/{folder_name}/train")) <= max(subjects):
        build_dataset(data_path, subjects, num_frames, shuffle=not video)
    
    train_dataset, val_dataset, test_dataset = None, None, None
    train_labels, val_labels, test_labels = [], [], []

    for i, subject in enumerate(subjects):
        subject_train = load_experiments(folder_name, subject, experiments, "train", device)
        subject_val = load_experiments(folder_name, subject, experiments, "validation", device)
        subject_test = load_experiments(folder_name, subject, experiments, "test", device)

        if train_dataset is None:
            train_dataset = subject_train
            val_dataset = subject_val
            test_dataset = subject_test
        else:
            train_dataset = torch.vstack((train_dataset, subject_train))
            val_dataset = torch.vstack((val_dataset, subject_val))
            test_dataset = torch.vstack((test_dataset, subject_test))
        
        train_labels.extend([i]*len(subject_train))
        val_labels.extend([i]*len(subject_val))
        test_labels.extend([i]*len(subject_test))
    
    train_labels, val_labels, test_labels = torch.tensor(train_labels, device=device, dtype=torch.int64), torch.tensor(val_labels, device=device, dtype=torch.int64), torch.tensor(test_labels, device=device, dtype=torch.int64)
    train_idx, val_idx, test_idx = batched_idxs(seed, [len(train_dataset), len(val_dataset), len(test_dataset)], batch_size)
    
    train_loader = [(train_dataset[batch], train_labels[batch]) for batch in train_idx]
    val_loader = [(val_dataset[batch], val_labels[batch]) for batch in val_idx]
    test_loader = [(test_dataset[batch], test_labels[batch]) for batch in test_idx]

    print(f"Train: {train_dataset.shape}")
    print(f"Validation: {val_dataset.shape}")
    print(f"Test: {test_dataset.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    return train_loader, val_loader, test_loader