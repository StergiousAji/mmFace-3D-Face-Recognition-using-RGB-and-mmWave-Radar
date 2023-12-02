import numpy as np
import matplotlib.pyplot as plt
import os

DATA_FOLDER = os.path.join("../../src/Soli/soli_realsense/data/")

def subject_samples(subject, experiments=[0, 1, 2]):
    base_paths = [os.path.join(DATA_FOLDER, f"{subject}/{subject}-{exp*5 + pose}_")  for exp in experiments for pose in range(5)]
    return [(f"{b}colour.npy", f"{b}depth.npy") for b in base_paths]

for subject in range(14):
    images = []
    for sample in subject_samples(str(subject)):
        images.append(np.load(sample[0]).astype(int)[0])
    images = np.array(images)
    print(images.shape)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')

    plt.savefig(f"images/faces/{subject}.png")