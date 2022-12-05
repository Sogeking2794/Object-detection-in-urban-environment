# Object Detection in an Urban Environment

## Project overview

This project demonstrates how convolutional neural networks may be trained on traffic datasets to detect objects. Transfer learning is used  with Tensorflow. It is crucial to recognize and catergorize every object in the image to comprehend a dynamic environment. To identify cars, other vehicles, pedestrians, and bicycles, we need to train a neural network. It compares the result from reference model without modification and model with augmented input images.

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

## Structure

As the first step Exploratory Data Analysis is done with ```Exploratory Data Analysis.ipynb```.  The training and validation datasets are split and a pre-trained model is loaded and trained. The model architecture is defined using config files ```pipeline.config```. Different data augmentation ```Explore Augmentations.ipynb```  and hyperparameters were experimented and the model performance on object detection is made better. Tensorboard is used to plot training, validation losses, precision, recall and mAP for small, medium and large sized images. The model prediction is then made into an animation to look at how well the model is able to detect objects in the environment.

### Data

The data used for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

### Experiments
The experiments folder is organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites. Use the provided Dockerfile and requirements in the [build directory](./build).

## Download and process the data

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to local machine. For this project,only a subset of the data provided is needed (for example, we do not need to use the Lidar data). Therefore, each file is downloaded and trimmed immediately. In `download_process.py`, the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). The `label_map.pbtxt` file is already provided

The script can be run using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. No need to make use of `gcloud` to download the images.


## Dataset Analysis

### Exploratory Data Analysis

This is to see the how the images and corresponding bounding boxes with classes looks like. A function to display images along with bounding boxes and classes is executed. The color coding of the boxes represent whether the object is a vehicle (Red), pedestrian (Blue) or a cyclist (Green). The number of objects in 20000 samples is plotted and compared. It is seen that most of the images has vehicle objects followed by pedestrians and small number of cylists. 

For the number of objects with count of frames, it is seen that there are lesser images with 0 vehicles and more images with 2-10 vehicles. The distribution decreases with increasing number of vehicles. There are upto 65 vehicles in an image. Is it also seen that there are large number of images with zero pedestrians and some images have upto 45 pedestrians. With the cyclists, it is rare to see image with them, atmost of 5 cyclists are seen in an image.


### Cross validation

### Training
#### Reference
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```
