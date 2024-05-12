# Adapting CNNs for Fisheye Cameras without Retraining

We introduce a pre-processing and fine-tunning free approach to adapting existing pre-trained networks to fisheye imagery. We present this in our paper:

Paper: [Adapting CNNs for Fisheye Cameras without Retraining](https://arxiv.org/abs/2404.08187)

Authors: [Ryan Griffiths](https://ryanbgriffiths.github.io), [Donald G. Dansereau](https://roboticimaging.org/)

Project Page: [roboticimaging.org/Projects/RectConv/](https://roboticimaging.org/Projects/RectConv/)

# Setup
### Environment
Get code and build docker image (requires docker to be installed):

    git clone https://github.com/RoboticImaging/RectConv.git
    cd RectifyConv
    docker build -t rect_conv .

### Dataset
Download the woodscape dataset [here](https://github.com/valeoai/WoodScape/tree/master)

### Checkpoints
Download checkpoints of pre-trained networks:
- [Deeplabv3_Resnet50](https://drive.google.com/file/d/1Ui_MPUJ0XWHCqg3c8_xelAVB81UEpnrf/view?usp=sharing)
- [Deeplabv3plus_Resnet101](https://drive.google.com/file/d/1A9hJAt5xUn8-qP9se3D0zjBqefmeQulH/view?usp=sharing)
- [FCN_Resnet50](https://drive.google.com/file/d/1dhmu3hM-LNC0AQDnT1hHRPmpPKrYSPzh/view?usp=sharing)
- [FCN_Resnet101](https://drive.google.com/file/d/1SDYDeoSBu5q-RrxKOLvQDtvULpNKvusm/view?usp=sharing)

# Run
Run the code inside container:

    docker run --rm \
    -v .:/workspace \
    --shm-size 4G \
    --gpus all \
    rect_conv \
    python run.py --model deeplabv3plus_resnet101 --data_path datasets/woodscape --model_checkpoints trained_models/deeplabv3plus_resnet101_cityscape.pth

*Note: to run the command above the dataset should be stored in a ./datasets directory and the checkpoints should be stored in a ./trained_models directory*

A *.devcontainer* folder is also provided if vscode used.

# Citation
If you find our work useful, please cite the below paper:  

    @article{griffiths2024adapting,
      title = {Adapting CNNs for Fisheye Cameras without Retraining},
      author = {Ryan Griffiths and Donald G. Dansereau},
      journal = {arXiv preprint arXiv:2404.08187},
      URL = {https://arxiv.org/abs/2404.08187},
      year = {2024},
      month = apr
    }
