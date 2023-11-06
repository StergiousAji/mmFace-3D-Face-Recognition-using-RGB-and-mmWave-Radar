## Meeting 1 (19/09/23) - 0:15

- Need to get data from scratch (ARC Physics Department equipment)
- Around 30-50 participants, get multiple angles (5-6) for test images
- Start reading for lit review
- Create a consent form for participants to agree to face scans

**Resources:**

- [3D Face Recognition: A Survey (2018)](https://hcis-journal.springeropen.com/articles/10.1186/s13673-018-0157-2)
- [3D Face Recognition: A Survey (2021)](https://arxiv.org/pdf/2108.11082.pdf)
- [Learning from Millions of 3D Scans for Large-scale 3D Face Recognition](https://arxiv.org/pdf/1711.05942.pdf)

**Face Recognition using RADAR:**

- [Face Recognition using mmWave RADAR imaging](https://ieeexplore.ieee.org/abstract/document/9701018/metrics#metrics)
- [Radar-Based Face Recognition: One-Shot Learning Approach](https://ieeexplore.ieee.org/abstract/document/9250469)
- [Face Verification Using mmWave Radar Sensor](https://www.semanticscholar.org/paper/Face-Verification-Using-mmWave-Radar-Sensor-Hof-Sanderovich/3ec5a616cc71d4a6a71aabd2c0b5a609b7d0eaea)

**Datasets**

- [EURECOM Kinect Face Dataset | rgb-d.eurecom.fr](http://rgb-d.eurecom.fr/)
- [Notre Dame CVRL (nd.edu)](https://cvrl.nd.edu/projects/data/#face-recognition-grand-challenge-frgc-v20-data-collection) (FRGC v2)

************************************************Questions For Next Week:************************************************

- Type of architecture appropriate: CNNs, RNNs, Transformers
- Combining RGB with RADAR??
- Possible 3D datasets to play around with, what technologies.

## Meeting 2 (26/09/23) - 0:25

- Device has RADAR, Range and RGB sensors
- RADAR gives sparse dataset while Range will give dense → want to try learn conversion from sparse to dense - more accurate → LATER, FIRST COMBINE RADAR WITH RGB
- 3 ways to combine 3D with 2D:
    - Combine at input stage so dataset will be $\text{Height} \times \text{Width} \times \text{Color} \times \text{Depth}$
    - Simple concatenation of 3D point cloud and 2D info
    - Combine at output stage to fine-tune prediction (ranking)
- Feature extraction of 2D - use pretrained models and fine-tune (ASK)
- Feature extraction of 3D - Point-Net / 1D Convolution (if point-cloud representation)
    
    $$
    \text{3D-Point-Cloud} = \begin{bmatrix}
    x_1 & y_1 & z_3 \\
    x_2 & y_2 & z_3 \\
    \vdots & \vdots & \vdots \\
    x_N & y_N & z_N
    \end{bmatrix}
    $$
    

### Meeting with Chaitanya

- Intel RealSense L515 (LiDAR + RGB + IR + IMU) and RADAR chip (1 Transmitter and 3 Receivers)
- Install Intel RealSense Viewer
- Chaitanya will provide code to receive data on Friday (DIDN’T HAPPEN)
- Ethics and Consent Form will be given before that :) (DIDN’T HAPPEN)

## Meeting 3 (03/10/23) - 0:15

- Waiting on Chaitanya’s code to start playing around with equipment.
- Established that project will be implemented in Python.
- 2D and 3D Face Analysis Project

https://github.com/deepinsight/insightface

## Meeting 4 (10/10/23) - 0:28

- Still waiting on Chaitanya
- Dataset will be made up of 4 differing variables:
    - **Expression:** BU-3/4DFE → 7 Expressions (MIGHT SCRAP)
    - **Pose:** 5 angles ($0\degree$, $\pm30\degree$, $\pm90\degree$)
    - **Light:** 2 lighting conditions (dim and regular)
    - **Occlusion:** 3 occluding objects (hat, glasses, mask)
- Estimate around 50 participants.
- Literature Review should be split into categories:
    - Dataset collection and variables (Plan)
    - 3 Modalities: Radar, RGB Camera, Range sensor
    - 3 2D-3D Fusion Techniques: Input, Concat, Output
    - Write out plan of next stages (estimated monthly)

### Meeting with Chaitanya (13/10/23)

- Received Ethics Consent Form Templates
    - **TODO:** Fill in and send for approval $\checkmark$
- C++ code initialises Radar sensor must compile with Makefile and run with `-p shortrange` for 20cm range. `simplelogger.cc`
    - MUST BE RUN IN LINUX/MAC
- Python code runs program to collect data. `testzmq.py`
- Read Chaitanya’s Previous Paper talks about details of Radar Sensor:
    - [mmSense: Detecting Concealed Weapons with a Miniature Radar Sensor](https://ieeexplore.ieee.org/abstract/document/10095884)
- ASK FOR A **TRIPOD $\checkmark$**

## Meeting 5 (17/10/23) - 0.10

- Progress Update:
    - Got Soli code from Chaitanya but cannot execute files on Ubuntu
    - Completed Ethics Forms will be finalising with Chaitanya
    - Finalised rough plan of data collection:
        - Only 1 expression: NEUTRAL
        - $5 \ \text{POSES} \ \times ((1 \ \text{NO-OCCLUSION}  \ \times \ 2 \ \text{LIGHTING}) + 3 \ \text{OCCLUSION}) = 25 \ \text{images/person}$
        - Around $1250$ images for $50$  participants
    - Start Interim Report draft next week

### Meeting with Chaitanya

- Got Soli running, need to install Anaconda to have all packages and Spyder.
    - **TODO:** Install ANACONDA $\checkmark$
    - Need to `pip3 install zmq`
- Meet Friday to get back devices and run full pipeline to get data
- Look at Google Soli papers:
    - [Soli: Ubiquitous Gesture Sensing with Millimeter Wave Radar](https://dl.acm.org/doi/pdf/10.1145/2897824.2925953)

### Meeting with Chaitanya (23/10/23)

- `testzmq.py` cannot capture real-time data :(
- Use `soli_nonrt/examples/main` (Sync RealSense and Radar to capture data simultaneously) and `soli_nonrt/examples/RS_no_p2go.py` on 2 separate terminals.
    - **TODO:** Configure `main.cc)` range and duration (MAY NOT NEED AS `simplelogger_zmq` WORKING)
    - **TODO:** Get RealSense USB detected on Ubuntu VM $\checkmark$
    

## Meeting 6 (24/10/23) - 0:12

- Use $80\%/10\%/10\%$ split for $\text{Train}/\text{Validation}/\text{Test}$
    - Or $68/16/16$ if ambitious
- Split by participants and use all poses for training.
- Talk about evaluation metrics in Interim Report
- These devices are thought out for an airport scenario.
- Got RealSense synced with Soli to get data by using `main` and `RS_rgb_new.py` *(modified)*:
    - **TODO:** Look at TODOs in `RS_rgb_new.py`
- Working on InsightFace with NBA players dataset

### Meeting with Chaitanya (30/10/23)

- Can simply feed the complex range-doppler (CRD) data into model. Maybe want to use `np.abs(crd_data)` first to get rid of complexity 😉
- Ethics will be done after 12th most likely
- Tripod will be given tomorrow from the ARC
- Use PyTorch to create CNN for radar recognition

## Meeting 7 (31/10/23) - 0.15

- Use PyTorch as most popular for building CNN
- Can directly feed complex data
- No prop department must use my own stuff for occlusion
- Just use protractor and ruler to measure angles for poses.

### Meeting with Chaitanya

- Received tripod. No mounting available just place cameras on top.
- Chaitanya emailing for access to a room in the ARC.

### Meeting with Chaitanya (02/11/23)

- Got access to the ARC and Room 371 (Doors lock themselves) (Access 8am-6pm)
- Setup sensors with tripod and marked tape for experiment.
- May need to look at lighting as currently face too dark (open curtain?)