# mmFace: 3D Face Recognition using RGB and Millimetre Wave Radar

The structure of the source code is organised as follows:
* `mmFace`: Directory holding all source code of models
    * `mmFace-hybrid.ipynb`: This holds the main **mmFace** hybrid model with the training loop and evaluation experiments.
    * `hybrid_dataset.py`: Auxiliary file holding functions to build and load the datasets into CUDA.
    * `neural_nets.py`: Auxiliary file holding PyTorch models and feature fusion functions.
    * `utils.py`: Auxiliary file holding useful functions including preprocessing help.
    * `results.ipynb`: Notebook to help aggregate evaluation results.
        * JSON files hold all evaluation results including the isolated experiment conditions, named accordingly.
* `soli_realsense`: Directory holding all source code for the data acquisition process.
    * `soli.cpp`: File to start running the Google's Soli chip and capture radar bursts once synchronised with the Intel Realsense camera.
    * `realsense.py`: File to capture data from the Intel Realsense camera, to be run in at the same time as the Soli.
    * `realsense_paper.py`: Copy of `realsense.py` with different configurations for capturing the fake faces.
    * `view_data.ipynb`: Notebook to help view the captured data.
    * `experiment.py`: Helper file holding all experiment information

----

[Python](https://www.python.org/downloads/) version `3.9.2` or later is required to run the source code. To check what version you have run this command on a command line terminal:
```cmd
> python --version
```

## Installation
The following commands will show the steps to clone the repository. First navigate to a suitable workspace directory and run:
```cmd
> git clone https://github.com/StergiousAji/mmFace-3D-Face-Recognition-using-RGB-and-mmWave-Radar.git
> cd mmFace-3D-Face-Recognition-using-RGB-and-mmWave-Radar\src
```

Finally, install the necessary packages from the `requirements.txt` file using the following command, and you're good to go:
```cmd
> pip install -r requirements.txt
```