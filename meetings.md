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

## Meeting 7 (31/10/23) - 0:15

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


## Meeting 8 (06/11/23) - 0:30

- Split by participants but metric will be the distance from unseen test data to (average) distribution of training embeddings.
    - Large distances correspond to poor model performance since faces should get mapped to same distribution.
    - Look at t-SNE (t-distributed Stochastic Neighbour Embedding) to classify distribution of high-D data preserving relationships between data (vs. PCA that preserves variance). Can be used to identify ***anomalies***

        [Introduction to t-SNE](https://www.datacamp.com/tutorial/introduction-t-sne)
- Starting Interim Report this week. Can show drafts to Hang for feedback.
- Once first data samples collected make presentation for outside perspective on project.


## Meeting 9 (14/11/23) - 0:15

- Start data collection this week.
- Can test accuracy on face recognition of test data vs. reference data (0, 1 if same person [similarity threshold])
- t-SNE can also group by different ages, genders if data is strong enough.


## Meeting 10 (21/11/23) - 0:23

- Ask Chaitanya about the transformation of the Soli data and hard-coded values in the `compute_rp` function.
- Add sample images into slides and high-level model architecture.
- Do not need to talk specifics about Radar technology.
- Feature fusion section should talk in specific relation to 3D face recognition.

### Meeting with Chaitanya (22/11/23)

- Experimented with capturing moving hand with Soli and CRD plot looks correct so fine to use64 chirps per burst when converting.
- a.u. means Arbitrary Units
- The 3 channels coming from the 3 receiver antennas of Soli.
- Data Acquisition presentation postponed to Monday 4th December. Can talk about model architecture next week.
- 2 more people available tomorrow 3 and half 3pm.


## Meeting 11 (27/11/23) - 0:20

- Reduce number of slides and don’t need to include CNN model architecture (black box).
- Talk to Chaitanya on how to process CRD data for the CNN model and design architecture (look at previous project on weapon detection).
- Talk about novelty of this project investigating common extreme occlusion scenarios while only paper previously did with cotton masks on 3 subjects.

### Meeting with Chaitanya (02/12/23)

- Discussed slides for data acquisition presentation on 04/12/23 (CANCELLED)
- Cut down slides to just 2, data samples shown clearly and model architecture good
- Talk about the different data fusion techniques and their feasibilities when showing model architecture.
- The CNN models created in the previous project were streamlined for real-time processing. Do not need to consider this with face recognition so can utilise the concepts within any of them.


## Meeting 12 (08/01/24) - 0:20

- Made a ResNet-based CNN although yields low accuracy model (15%)
- Need to improve maybe by treating as video data during training.
- Increasing/Reducing network complexity
- Fall-back idea to use Radar for pure liveness check since Insightface able to achieve 100%
    - Use for binary classification of real 3D face vs. 2D printed face image
- Ask Chaitanya about specifics of mmSense architecture and format/pre-processing of CRD data
- Ask Chaitanya about Angle of Arrival (AOA) data and if that is useful.


## Meeting 13 (15/01/24) - 0:20

- Focus on Liveness Binary Classification since ResNet18 finetuning overfitting
- Can try ResNet50 and separately 3D video data approach → Need to reformat data loaders for this.
- Could try data augmentation within data loaders to inherently dataset size

### Meeting with Chaitanya (Multiple)

- Too much variation in the data for the model to learn without overfitting.
- Can achieve 70-72% on subset of 4 subjects and 45% on 6 with simple CNN.
- Proximity of face to sensor is the issue need to be much closer than 20cm for accurate data.


## Meeting 14 (22/01/24) - 0:30

- Didn’t get chance to gather liveness data, will gather 2D data on 10 subjects at least (frontal poses).
- Need to separate out classifiers to combine feature vectors also do the same for the InsightFace model to see how accurate it is at identifying liveness (need to ensure paper is not seen).
- For InsightFace+mmFace need some way of combining two models into a single class to take in both modalities and output [subject, liveness?] predictions.


## Meeting 15 (29/01/24) - 0:30

- Need to make a custom model that combines 2 parameterised loss functions working on the 2 feature vectors separately.
    - Model takes in Radar input and 2D feature vector and tunes the radar feature extraction and finally has 2 FC layers to output subject? and liveness?
    - Can redefine forward pass to take 2 inputs and give 2 outputs.
    - Loss function combines the loss from the subject and liveness classifications


## Meeting 16 (05/02/24) - 0:30

- Try zero-shot classification by trying to predict unseen classes/subjects.


## Meeting 17 (12/02/24) - 0:20

- During testing, extract features of single reference inputs for each class and use cosine similarity to test accuracy of embeddings.
- Use data augmentation (2 flips) on both inputs to increase dataset diversity


## Meeting 18 (19/02/24) - 0:20

- Getting 75-80% face zero-shot performance on 4 subjects :)
- Use different feature fusions to identify if better than concatenation:
    - Addition
    - Convolution-based
    - Multi-head Attention Mechanism
- Compare with more complex feature extraction for radar ARDs (ResNet?)
- Compare with only RGB vs. only Radar vs. both


## Meeting 19 (26/02/24) - 0:25

- Explain ROC curve in terms of Sensitivity (TPR) and 1-Specificity (FPR) to explain dip for concatenation
- Should weight accuracy more highly when comparing models since AUC showing positive results when low accuracy
- Separate t-SNE for different poses, lighting conditions, occlusions and liveness to show clusters for 4 subjects
- See how t-SNE responds to the raw inputs compared to embedded if worth talking about.
- Still yet to complete Multi-head Attention


## Meeting 20 (04/03/24) - 0:15

- Show specific experiment t-SNE visualisations for best performing feature fusions
- Compare best performing models regardless of number of epochs trained
- Cancelled next week meeting due to deadlines


## Meeting 21 (18/03/24) - 0:10

- Started writing final paper
- Interim report grade not released but expect A band
- Final presentation date will be notified sooner to date.


## Meeting 22 (25/03/24) - 0:10

- Should change Research Aims section in Intro to Contributions and summarise achievements of project.
- Final presentation date around week 3 of April since deadline extended to 30th.