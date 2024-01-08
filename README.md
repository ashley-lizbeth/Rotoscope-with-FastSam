This project is a rotoscoping tool to automatically rotoscope a video file. It uses [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) to perform object segmentation on every frame of the video, and draws the contour creating a rotoscope. Right now, it is quite inneficient because it has to recalculate every object in the scene every single frame, and there is much room for improvement, like using a different techology.

## Installation instructions
Clone the github repository:

```shell
https://github.com/ashley-lizbeth/Rotoscope-with-FastSam.git
```

As per FastSAM's instructions, the code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
conda create -n FastSAM python=3.9
conda activate Rotoscope-with-FastSam
```

Install the packages:

```shell
cd Rotoscope-with-FastSam
pip install -r requirements.txt
```

Install CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```

## Running the program

First download a [model checkpoint](https://github.com/CASIA-IVA-Lab/FastSAM/tree/main#model-checkpoints). If the link doesn't work, visit the [Getting Started](https://github.com/CASIA-IVA-Lab/FastSAM/tree/main#-getting-started) section of the FastSAM repo for the latest model. Then, create a `weights/` folder and place the model inside.
<br><br>
Then, open main.py and change the variable [VIDEO_PATH](https://github.com/ashley-lizbeth/Rotoscope-with-FastSam/blob/8cf6d4f7401173c2d52497164deefa485cc6228d/main.py#L13) to the path of your video, preferably creating a folder named `input/` and placing it inside.

```python
VIDEO_PATH = "input/your_video.mp4"
```

Finally, you can run main.py.

```shell
python3 ./main.py
```

And wait!
