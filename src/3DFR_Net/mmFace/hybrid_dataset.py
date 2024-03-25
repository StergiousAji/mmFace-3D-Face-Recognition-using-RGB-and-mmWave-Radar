from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import get_crd_data
from tqdm import tqdm
import torch
import json
import os
import numpy as np

# AUXILLIARY FUNCTIONS AND HYBRID DATASET
def get_ard(path, subject, experiment, num_frames):
        with open(f"{path}/{subject}/{subject}-{experiment}_radar.json", 'r') as f:
            ard = np.abs(get_crd_data(json.load(f), num_chirps_per_burst=16))[:num_frames].astype(np.float32)
            # Return shuffled frames
            return ard[np.random.permutation(len(ard))]

def get_rgb_emb(subject, experiment, real, fake):
    if subject < 90:
        data = real
    else:
        data = fake
        subject = int(str(subject)[1:])

    embs = data[subject, experiment]
    # Filter NaNs and Undetected Faces
    filtered = embs[~np.isnan(embs).any(axis=1)]
    return filtered

class HybridDataset(Dataset):
    def __init__(self, radar_data, rgb_embs, subject_labels, liveness_labels, split):
        self.radar_data = radar_data
        self.rgb_embs = rgb_embs
        self.subject_labels = subject_labels
        self.liveness_labels = liveness_labels
        self.split = split

    def __len__(self):
        return len(self.radar_data)
    
    def __getitem__(self, idx):
        return self.radar_data[idx], self.rgb_embs[idx], self.subject_labels[idx], self.liveness_labels[idx]


# PLAIN DATASET WITH [TRAIN, VAL, TEST]
def build_dataset(raw_path, num_subjects, num_experiments=15, train_split=22/28, test_split=3/28, device="cuda", seed=24):
    np.random.seed(seed)
    subjects = list(range(num_subjects)) + [int(f"9{s}") for s in range(num_subjects)]
    experiments = range(num_experiments)
    val_split_end = train_split + test_split

    data = {"radar": [[], [], []], "rgb_embs": [[], [], []]}
    labels = {"subject": [[], [], []], "liveness": [[], [], []]}

    # Subjects x Experiments x Frames x Embedding_Size *(21 x 15 x 10 x 512)
    real_rgb = np.load("data/InsightFace_embs/real_insightface_embs.npy")
    fake_rgb = np.load("data/InsightFace_embs/fake_insightface_embs.npy")

    for subject in tqdm(subjects):
        num_frames = 250 if subject < 90 else 74
        live = 0 if subject < 90 else 1
        experiments = range(num_experiments) if subject < 90 else [0, 5, 10]
        sub_label = subject if subject < 90 else int(str(subject)[1:])
        for experiment in experiments:
            rgb_emb = get_rgb_emb(subject, experiment, real_rgb, fake_rgb)
            # Discard experiments with no RGB embeddings
            if len(rgb_emb) < 1:
                continue
            
            exp_ard = get_ard(raw_path, subject, experiment, num_frames)

            n = 15 if subject < 90 else len(exp_ard)

            # Duplicate embeddings until n frames to match size of ARD dataset
            rgb_emb = np.tile(rgb_emb, (n//len(rgb_emb) + 1, 1))[:n]
            rgb_train = rgb_emb[:int(n*train_split)]
            rgb_val = rgb_emb[int(n*train_split):int(n*val_split_end)]
            rgb_test = rgb_emb[int(n*val_split_end):n]
            
            exp_train = exp_ard[:int(n*train_split)]
            exp_val = exp_ard[int(n*train_split):int(n*val_split_end)]
            exp_test = exp_ard[int(n*val_split_end):n]
            del rgb_emb, exp_ard

            data["radar"][0].append(exp_train)
            data["radar"][1].append(exp_val)
            data["radar"][2].append(exp_test)

            data["rgb_embs"][0].append(rgb_train)
            data["rgb_embs"][1].append(rgb_val)
            data["rgb_embs"][2].append(rgb_test)

            labels["subject"][0].append([sub_label]*len(exp_train))
            labels["subject"][1].append([sub_label]*len(exp_val))
            labels["subject"][2].append([sub_label]*len(exp_test))

            labels["liveness"][0].append([live]*len(exp_train))
            labels["liveness"][1].append([live]*len(exp_val))
            labels["liveness"][2].append([live]*len(exp_test))
    del real_rgb, fake_rgb

    data["radar"][0] = np.concatenate(data["radar"][0])
    data["radar"][1] = np.concatenate(data["radar"][1])
    data["radar"][2] = np.concatenate(data["radar"][2])

    data["rgb_embs"][0] = np.concatenate(data["rgb_embs"][0])
    data["rgb_embs"][1] = np.concatenate(data["rgb_embs"][1])
    data["rgb_embs"][2] = np.concatenate(data["rgb_embs"][2])

    labels["subject"][0] = np.concatenate(labels["subject"][0])
    labels["subject"][1] = np.concatenate(labels["subject"][1])
    labels["subject"][2] = np.concatenate(labels["subject"][2])

    labels["liveness"][0] = np.concatenate(labels["liveness"][0])
    labels["liveness"][1] = np.concatenate(labels["liveness"][1])
    labels["liveness"][2] = np.concatenate(labels["liveness"][2])
    
    print(len(data["radar"][0]), len(data["radar"][1]), len(data["radar"][2]))
    print(len(data["rgb_embs"][0]), len(data["rgb_embs"][1]), len(data["rgb_embs"][2]))
    
    torch.save(data, "data/hybrid/dataset.pt")
    torch.save(labels, "data/hybrid/labels.pt")

def load_dataset(raw_path, num_subjects, batch_size=128, device="cuda", seed=24):
    if not os.path.exists("data/hybrid/dataset.pt"):
        build_dataset(raw_path, num_subjects)
    
    data = torch.load("data/hybrid/dataset.pt")
    labels = torch.load("data/hybrid/labels.pt")

    train_radar = torch.tensor(data["radar"][0], device=device)
    val_radar = torch.tensor(data["radar"][1], device=device)
    test_radar = torch.tensor(data["radar"][2], device=device)

    train_rgb = torch.tensor(data["rgb_embs"][0], device=device, dtype=torch.float32)
    val_rgb = torch.tensor(data["rgb_embs"][1], device=device, dtype=torch.float32)
    test_rgb = torch.tensor(data["rgb_embs"][2], device=device, dtype=torch.float32)

    train_labels_s = torch.tensor(labels["subject"][0], device=device, dtype=torch.int64)
    val_labels_s = torch.tensor(labels["subject"][1], device=device, dtype=torch.int64)
    test_labels_s = torch.tensor(labels["subject"][2], device=device, dtype=torch.int64)

    train_labels_l = torch.tensor(labels["liveness"][0], device=device, dtype=torch.int64)
    val_labels_l = torch.tensor(labels["liveness"][1], device=device, dtype=torch.int64)
    test_labels_l = torch.tensor(labels["liveness"][2], device=device, dtype=torch.int64)

    del data, labels

    print(f"Train (Radar): {train_radar.shape}")
    print(f"Train (RGB Embeddings): {train_rgb.shape}")
    print(f"Validation (Radar): {val_radar.shape}")
    print(f"Validation (RGB Embeddings): {val_rgb.shape}")
    print(f"Test (Radar): {test_radar.shape}")
    print(f"Test (RGB Embeddings): {test_rgb.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    train_dataset = HybridDataset(train_radar, train_rgb, train_labels_s, train_labels_l, "train")
    val_dataset = HybridDataset(val_radar, val_rgb, val_labels_s, val_labels_l, "validation")
    test_dataset = HybridDataset(test_radar, test_rgb, test_labels_s, test_labels_l, "test")

    np.random.seed(seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(train_dataset))))
    val_loader = DataLoader(val_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(test_dataset))))

    return train_loader, val_loader, test_loader




from collections import defaultdict
import pickle

# PLAIN DATASET WITH PROPORTIONATE SPLITTING OF EXPERIMENTS
def build_dataset_subject(raw_path, num_subjects, num_experiments=15):
    subjects = list(range(num_subjects)) + [int(f"9{s}") for s in range(num_subjects)]
    print(subjects)

    data = {"radar": defaultdict(list), "rgb_embs": defaultdict(list)}

    # Subjects x Experiments x Frames x Embedding_Size *(21 x 15 x 10 x 512)
    real_rgb = np.load("data/InsightFace_embs/real_insightface_embs.npy")
    fake_rgb = np.load("data/InsightFace_embs/fake_insightface_embs.npy")

    for subject in tqdm(subjects):
        num_frames = 15 if subject < 90 else 74
        experiments = range(num_experiments) if subject < 90 else [0, 5, 10]
        sub = subject if subject < 90 else int(str(subject)[1:]) + num_subjects
        for experiment in experiments:
            rgb_emb = get_rgb_emb(subject, experiment, real_rgb, fake_rgb)
            # Discard experiments with no RGB embeddings at all
            if len(rgb_emb) < 1:
                print(subject, experiment)
                continue

            exp_ard = get_ard(raw_path, subject, experiment, num_frames)

            # Duplicate embeddings until n frames to match size of ARD dataset
            rgb_emb = np.tile(rgb_emb, (len(exp_ard)//len(rgb_emb) + 1, 1))[:len(exp_ard)]
            
            data["radar"][sub].append(exp_ard)
            data["rgb_embs"][sub].append(rgb_emb)
        
        data["radar"][sub] = np.concatenate(data["radar"][sub])
        data["rgb_embs"][sub] = np.concatenate(data["rgb_embs"][sub])

    del real_rgb, fake_rgb

    data["radar"] = np.array(list(data["radar"].values()), dtype=object)
    data["rgb_embs"] = np.array(list(data["rgb_embs"].values()), dtype=object)

    print(len(data["radar"]), len(data["rgb_embs"]))

    with open("data/hybrid-by_subject/dataset.pickle", 'wb') as f:
        pickle.dump(data, f)


def load_dataset_subject(raw_path, num_subjects, batch_size=128, train_split=17/21, test_split=2/21, device="cuda", seed=54):
    if not os.path.exists("data/hybrid-by_subject/dataset.pickle"):
        build_dataset_subject(raw_path, num_subjects)

    with open("data/hybrid-by_subject/dataset.pickle", 'rb') as f:
        data = pickle.load(f)

    val_split_end = train_split + test_split
    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(num_subjects).tolist()
    # Add fake subjects to the indexes
    train_idx = shuffled_subjects[:int(train_split*len(shuffled_subjects))]
    train_idx.extend([num_subjects + s for s in train_idx])
    val_idx = shuffled_subjects[int(train_split*len(shuffled_subjects)):int(val_split_end*len(shuffled_subjects))]
    val_idx.extend([num_subjects + s for s in val_idx])
    test_idx = shuffled_subjects[int(val_split_end*len(shuffled_subjects)):]
    test_idx.extend([num_subjects + s for s in test_idx])

    print(train_idx)
    print(val_idx)
    print(test_idx)

    train_radar = torch.tensor(np.concatenate(data["radar"][train_idx]), device=device)
    val_radar = torch.tensor(np.concatenate(data["radar"][val_idx]), device=device)
    test_radar = torch.tensor(np.concatenate(data["radar"][test_idx]), device=device)

    train_rgb = torch.tensor(np.concatenate(data["rgb_embs"][train_idx]), device=device, dtype=torch.float32)
    val_rgb = torch.tensor(np.concatenate(data["rgb_embs"][val_idx]), device=device, dtype=torch.float32)
    test_rgb = torch.tensor(np.concatenate(data["rgb_embs"][test_idx]), device=device, dtype=torch.float32)

    train_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    val_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in val_idx]), device=device, dtype=torch.int64)
    test_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    train_labels_l = torch.tensor(np.concatenate([[0 if sub < num_subjects else 1]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    val_labels_l = torch.tensor(np.concatenate([[0 if sub < num_subjects else 1]*len(data["radar"][sub]) for sub in val_idx]), device=device, dtype=torch.int64)
    test_labels_l = torch.tensor(np.concatenate([[0 if sub < num_subjects else 1]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    print(f"Train (Radar): {train_radar.shape}")
    print(f"Train (RGB Embeddings): {train_rgb.shape}")
    print(f"Validation (Radar): {val_radar.shape}")
    print(f"Validation (RGB Embeddings): {val_rgb.shape}")
    print(f"Test (Radar): {test_radar.shape}")
    print(f"Test (RGB Embeddings): {test_rgb.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    train_dataset = HybridDataset(train_radar, train_rgb, train_labels_s, train_labels_l, "train")
    val_dataset = HybridDataset(val_radar, val_rgb, val_labels_s, val_labels_l, "validation")
    test_dataset = HybridDataset(test_radar, test_rgb, test_labels_s, test_labels_l, "test")

    np.random.seed(seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(train_dataset))))
    val_loader = DataLoader(val_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(test_dataset))))

    return train_loader, val_loader, test_loader





# DATA WITH AUGMENTATIONS AND PROPORTIONATE EXPERIMENT SPLITTING
import torch
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

BASE_DIR = os.getcwd()
det_model_path = os.path.join(BASE_DIR, "../models", "buffalo_l", "det_10g.onnx")
rec_model_path = os.path.join(BASE_DIR, "../models", "buffalo_l", "w600k_r50.onnx")
det_model = model_zoo.get_model(det_model_path)
rec_model = model_zoo.get_model(rec_model_path)
det_model.prepare(ctx_id=0, input_size=(480, 640), det_thres=0.5)

def augment(data, h=2, v=1):
    return {"none": data, "horizontal_flip": np.flip(data, axis=h), "vertical_flip": np.flip(data, axis=v)}

def get_ard_aug(path, subject, experiment, num_frames, transform):
    with open(f"{path}/{subject}/{subject}-{experiment}_radar.json", 'r') as f:
        ard = np.abs(get_crd_data(json.load(f), num_chirps_per_burst=16))[:num_frames].astype(np.float32)
        return augment(ard, 3, 2)[transform]

def get_rgb_emb_aug(path, subject, experiment, transform, undetected):
    img_path = rf"{path}\{subject}\{subject}-{experiment}_colour.npy"
    rgb_frames = np.load(img_path).astype(np.float32)
    frames = augment(rgb_frames[..., ::-1])[transform]

    embeddings = np.full((10, 512), np.nan)
    for frame, img in enumerate(frames):
        # Unable to detect vertical faces
        bboxes, kpss = det_model.detect(img, max_num=0, metric="default")
        if len(bboxes) != 1:
            undetected.add((img_path.split('\\')[-1], frame, transform))
            continue
        face = Face(bbox=bboxes[0, :4], kps=kpss[0], det_score=bboxes[0, 4])
        rec_model.get(img, face)
        embeddings[frame] = face.normed_embedding

    # Filter nan embeddings
    return embeddings[~np.isnan(embeddings).any(axis=1)]


def build_aug_dataset(path, num_subjects, num_experiments=15, seed=38):
    np.random.seed(seed)
    subjects = list(range(num_subjects)) + [int(f"9{s}") for s in range(num_subjects)]
    print(subjects)

    data = {"radar": defaultdict(list), "rgb_embs": defaultdict(list)}

    undetected_faces = set()

    for subject in tqdm(subjects):
        num_frames = 15 if subject < 90 else 74
        experiments = range(num_experiments) if subject < 90 else [0, 5, 10]
        sub = subject if subject < 90 else int(str(subject)[1:]) + num_subjects
        for experiment in experiments:
            for transform in ["none", "horizontal_flip", "vertical_flip"]:
                rgb_emb = get_rgb_emb_aug(path, subject, experiment, transform, undetected_faces)
                # Discard experiments with no RGB embeddings at all
                if len(rgb_emb) < 1:
                    print(subject, experiment, transform)
                    continue
                
                ard = get_ard_aug(path, subject, experiment, num_frames, transform)

                # Duplicate embeddings until n frames to match size of ARD dataset
                rgb_emb = np.tile(rgb_emb, (len(ard)//len(rgb_emb) + 1, 1))[:len(ard)]

                data["radar"][sub].append(ard)
                data["rgb_embs"][sub].append(rgb_emb)
        
        data["radar"][sub] = np.concatenate(data["radar"][sub])
        data["rgb_embs"][sub] = np.concatenate(data["rgb_embs"][sub])

        shuffled = np.random.permutation(len(data["radar"][sub]))

        data["radar"][sub] = data["radar"][sub][shuffled]
        data["rgb_embs"][sub] = data["rgb_embs"][sub][shuffled]


    print(data["radar"][0].shape, data["rgb_embs"][0].shape)

    with open("data/hybrid-augmented/dataset.pickle", 'wb') as f:
        pickle.dump(data, f)
    
    if len(undetected_faces) > 0:
        with open('data/hybrid-augmented/undetected_faces.json', 'a', encoding='utf-8') as g:
            json.dump(list(undetected_faces), g, ensure_ascii=False, indent=4)


def load_aug_dataset(path, num_subjects, batch_size=128, train_split=0.8, test_split=0.1, device="cuda", seed=54):
    if not os.path.exists("data/hybrid-augmented/dataset.pickle"):
        build_aug_dataset(path, num_subjects)

    np.random.seed(seed)
    with open("data/hybrid-augmented/dataset.pickle", 'rb') as f:
        data = pickle.load(f)
    
    # List of concat data grouped by subject (REAL: 15 frames per exp, FAKE: 74 frames per exp)
    data["radar"] = np.array(list(data["radar"].values()), dtype=object)
    data["rgb_embs"] = np.array(list(data["rgb_embs"].values()), dtype=object)

    val_split_end = train_split + test_split

    train_radar, val_radar, test_radar = [], [], []
    train_rgb, val_rgb, test_rgb = [], [], []

    train_labels_s, val_labels_s, test_labels_s = [], [], []
    train_labels_l, val_labels_l, test_labels_l = [], [], []

    for sub in range(len(data["radar"])):
        frames = 15 if sub < num_subjects else 74
        sub_label = sub if sub < num_subjects else sub - num_subjects
        live = 0 if sub < num_subjects else 1
        for i in range(0, len(data["radar"][sub]), frames):
            shuffled_idxs = np.random.permutation(frames)
            radar_exp = data["radar"][sub][i:i+frames][shuffled_idxs]
            rgb_exp = data["rgb_embs"][sub][i:i+frames][shuffled_idxs]

            radar_exp_train = radar_exp[:int(train_split*frames)]
            radar_exp_val = radar_exp[int(train_split*frames):int(val_split_end*frames)]
            radar_exp_test = radar_exp[int(val_split_end*frames):]

            train_radar.append(radar_exp_train)
            val_radar.append(radar_exp_val)
            test_radar.append(radar_exp_test)

            train_rgb.append(rgb_exp[:int(train_split*frames)])
            val_rgb.append(rgb_exp[int(train_split*frames):int(val_split_end*frames)])
            test_rgb.append(rgb_exp[int(val_split_end*frames):])

            train_labels_s.append([sub_label]*len(radar_exp_train))
            val_labels_s.append([sub_label]*len(radar_exp_val))
            test_labels_s.append([sub_label]*len(radar_exp_test))

            train_labels_l.append([live]*len(radar_exp_train))
            val_labels_l.append([live]*len(radar_exp_val))
            test_labels_l.append([live]*len(radar_exp_test))


    train_radar = torch.tensor(np.concatenate(train_radar), device=device)
    val_radar = torch.tensor(np.concatenate(val_radar), device=device)
    test_radar = torch.tensor(np.concatenate(test_radar), device=device)

    train_rgb = torch.tensor(np.concatenate(train_rgb), device=device, dtype=torch.float32)
    val_rgb = torch.tensor(np.concatenate(val_rgb), device=device, dtype=torch.float32)
    test_rgb = torch.tensor(np.concatenate(test_rgb), device=device, dtype=torch.float32)

    train_labels_s = torch.tensor(np.concatenate(train_labels_s), device=device, dtype=torch.int64)
    val_labels_s = torch.tensor(np.concatenate(val_labels_s), device=device, dtype=torch.int64)
    test_labels_s = torch.tensor(np.concatenate(test_labels_s), device=device, dtype=torch.int64)

    train_labels_l = torch.tensor(np.concatenate(train_labels_l), device=device, dtype=torch.int64)
    val_labels_l = torch.tensor(np.concatenate(val_labels_l), device=device, dtype=torch.int64)
    test_labels_l = torch.tensor(np.concatenate(test_labels_l), device=device, dtype=torch.int64)

    print(f"Train (Radar): {train_radar.shape}")
    print(f"Train (RGB Embeddings): {train_rgb.shape}")
    print(f"Validation (Radar): {val_radar.shape}")
    print(f"Validation (RGB Embeddings): {val_rgb.shape}")
    print(f"Test (Radar): {test_radar.shape}")
    print(f"Test (RGB Embeddings): {test_rgb.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    train_dataset = HybridDataset(train_radar, train_rgb, train_labels_s, train_labels_l, "train")
    val_dataset = HybridDataset(val_radar, val_rgb, val_labels_s, val_labels_l, "validation")
    test_dataset = HybridDataset(test_radar, test_rgb, test_labels_s, test_labels_l, "test")

    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(train_dataset))))
    val_loader = DataLoader(val_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(val_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(test_dataset))))

    return train_loader, val_loader, test_loader


def load_aug_dataset_subject(path, num_subjects, batch_size=128, train_split=17/21, device="cuda", seed=54):
    if not os.path.exists("data/hybrid-augmented/dataset.pickle"):
        build_aug_dataset(path, num_subjects)

    with open("data/hybrid-augmented/dataset.pickle", 'rb') as f:
        data = pickle.load(f)
    
    data["radar"] = np.array(list(data["radar"].values()), dtype=object)
    data["rgb_embs"] = np.array(list(data["rgb_embs"].values()), dtype=object)

    print(len(data["radar"]), len(data["rgb_embs"]))

    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(num_subjects).tolist()
    train_idx = shuffled_subjects[:int(train_split*len(shuffled_subjects))]
    # Add fake subjects to the indexes
    train_idx.extend([num_subjects + s for s in train_idx]) 

    test_idx = shuffled_subjects[int(train_split*len(shuffled_subjects)):]
    test_idx.extend([num_subjects + s for s in test_idx])

    print(train_idx)
    print(test_idx)

    train_radar = torch.tensor(np.concatenate(data["radar"][train_idx]), device=device)
    test_radar = torch.tensor(np.concatenate(data["radar"][test_idx]), device=device)

    train_rgb = torch.tensor(np.concatenate(data["rgb_embs"][train_idx]), device=device, dtype=torch.float32)
    test_rgb = torch.tensor(np.concatenate(data["rgb_embs"][test_idx]), device=device, dtype=torch.float32)

    train_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    test_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    train_labels_l = torch.tensor(np.concatenate([[1 if sub < num_subjects else 0]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    test_labels_l = torch.tensor(np.concatenate([[1 if sub < num_subjects else 0]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    print(f"Train (Radar): {train_radar.shape}")
    print(f"Train (RGB Embeddings): {train_rgb.shape}")
    print(f"Test (Radar): {test_radar.shape}")
    print(f"Test (RGB Embeddings): {test_rgb.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    train_dataset = HybridDataset(train_radar, train_rgb, train_labels_s, train_labels_l, "train")
    test_dataset = HybridDataset(test_radar, test_rgb, test_labels_s, test_labels_l, "test")

    np.random.seed(seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(train_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(test_dataset))))

    return train_loader, test_loader, test_idx



# TRACK EXPERIMENTS
def build_aug_dataset_exp(path, num_subjects, num_experiments=15, seed=38):
    np.random.seed(seed)
    subjects = list(range(num_subjects)) + [int(f"9{s}") for s in range(num_subjects)]
    print(subjects)

    data = {"radar": defaultdict(list), "rgb_embs": defaultdict(list), "experiments": defaultdict(list)}

    undetected_faces = set()

    for subject in tqdm(subjects):
        num_frames = 15 if subject < 90 else 74
        experiments = range(num_experiments) if subject < 90 else [0, 5, 10]
        sub = subject if subject < 90 else int(str(subject)[1:]) + num_subjects
        for experiment in experiments:
            for transform in ["none", "horizontal_flip", "vertical_flip"]:
                rgb_emb = get_rgb_emb_aug(path, subject, experiment, transform, undetected_faces)
                # Discard experiments with no RGB embeddings at all
                if len(rgb_emb) < 1:
                    print(subject, experiment, transform)
                    continue
                
                ard = get_ard_aug(path, subject, experiment, num_frames, transform)

                # Duplicate embeddings until n frames to match size of ARD dataset
                rgb_emb = np.tile(rgb_emb, (len(ard)//len(rgb_emb) + 1, 1))[:len(ard)]

                data["radar"][sub].append(ard)
                data["rgb_embs"][sub].append(rgb_emb)
                data["experiments"][sub].append([experiment for _ in range(num_frames)])
        
        data["radar"][sub] = np.concatenate(data["radar"][sub])

        shuffled = np.random.permutation(len(data["radar"][sub]))
        data["radar"][sub] = data["radar"][sub][shuffled]
        data["rgb_embs"][sub] = np.concatenate(data["rgb_embs"][sub])[shuffled]
        data["experiments"][sub] = np.concatenate(data["experiments"][sub])[shuffled]


    print(data["radar"][0].shape, data["rgb_embs"][0].shape, data["experiments"][0].shape)

    with open("data/hybrid-augmented-exp/dataset.pickle", 'wb') as f:
        pickle.dump(data, f)
    
    if len(undetected_faces) > 0:
        with open('data/hybrid-augmented-exp/undetected_faces.json', 'a', encoding='utf-8') as g:
            json.dump(list(undetected_faces), g, ensure_ascii=False, indent=4)


def load_aug_dataset_subject_exp(path, num_subjects, batch_size=128, train_split=17/21, device="cuda", seed=54):
    if not os.path.exists("data/hybrid-augmented-exp/dataset.pickle"):
        build_aug_dataset_exp(path, num_subjects)

    with open("data/hybrid-augmented-exp/dataset.pickle", 'rb') as f:
        data = pickle.load(f)
    
    data["radar"] = np.array(list(data["radar"].values()), dtype=object)
    data["rgb_embs"] = np.array(list(data["rgb_embs"].values()), dtype=object)
    data["experiments"] = np.array(list(data["experiments"].values()), dtype=object)

    print(len(data["radar"]), len(data["rgb_embs"]), len(data["experiments"]))

    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(num_subjects).tolist()
    train_idx = shuffled_subjects[:int(train_split*len(shuffled_subjects))]
    # Add fake subjects to the indexes
    train_idx.extend([num_subjects + s for s in train_idx]) 

    test_idx = shuffled_subjects[int(train_split*len(shuffled_subjects)):]
    test_idx.extend([num_subjects + s for s in test_idx])

    print(train_idx)
    print(test_idx)

    train_radar = torch.tensor(np.concatenate(data["radar"][train_idx]), device=device)
    test_radar = torch.tensor(np.concatenate(data["radar"][test_idx]), device=device)

    train_rgb = torch.tensor(np.concatenate(data["rgb_embs"][train_idx]), device=device, dtype=torch.float32)
    test_rgb = torch.tensor(np.concatenate(data["rgb_embs"][test_idx]), device=device, dtype=torch.float32)

    train_exp = np.concatenate(data["experiments"][train_idx])
    test_exp = np.concatenate(data["experiments"][test_idx])

    train_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    test_labels_s = torch.tensor(np.concatenate([[sub if sub < num_subjects else sub-num_subjects]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    train_labels_l = torch.tensor(np.concatenate([[1 if sub < num_subjects else 0]*len(data["radar"][sub]) for sub in train_idx]), device=device, dtype=torch.int64)
    test_labels_l = torch.tensor(np.concatenate([[1 if sub < num_subjects else 0]*len(data["radar"][sub]) for sub in test_idx]), device=device, dtype=torch.int64)

    print(f"Train (Radar): {train_radar.shape}")
    print(f"Train (RGB Embeddings): {train_rgb.shape}")
    print(f"Test (Radar): {test_radar.shape}")
    print(f"Test (RGB Embeddings): {test_rgb.shape}")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    train_dataset = HybridDataset(train_radar, train_rgb, train_labels_s, train_labels_l, "train")
    test_dataset = HybridDataset(test_radar, test_rgb, test_labels_s, test_labels_l, "test")

    np.random.seed(seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(train_dataset))))
    test_loader = DataLoader(test_dataset, batch_size, sampler=SubsetRandomSampler(np.random.permutation(len(test_dataset))))

    return train_loader, test_loader, test_idx, test_exp