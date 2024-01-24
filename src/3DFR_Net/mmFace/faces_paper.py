import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_FOLDER = "../../../Soli/soli_realsense/data"

for subject in tqdm(range(21)):
    for exp in [0, 5, 10]:
        frame = np.load(f"{DATA_FOLDER}/{subject}/{subject}-{exp}_colour.npy").astype(np.uint8)[0]
        Image.fromarray(frame).save(f"{subject}_{exp}.jpg")