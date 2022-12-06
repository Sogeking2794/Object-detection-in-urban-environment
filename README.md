# Object Detection in an Urban Environment

## Project overview

This project demonstrates how convolutional neural networks may be trained on traffic datasets to detect objects. Transfer learning is used  with Tensorflow. It is crucial to recognize and catergorize every object in the image to comprehend a dynamic environment. To identify cars, other vehicles, pedestrians, and bicycles, we need to train a neural network. It compares the result from reference model without modification and model with augmented input images.

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

## Structure

As the first step Exploratory Data Analysis is done with ```Exploratory Data Analysis.ipynb```.  The training and validation datasets are split and a pre-trained model is loaded and trained. The model architecture is defined using config files ```pipeline.config```. Different data augmentation ```Explore Augmentations.ipynb```  and hyperparameters were experimented and the model performance on object detection is made better. Tensorboard is used to plot training, validation losses, precision and recall for small, medium and large sized images. The model prediction is then made into an animation to look at how well the model is able to detect objects in the environment.

### Data

The data used for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: 87 files to train
    - val: 10 files to validate
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

This is to see the how the images and corresponding bounding boxes with classes looks like. A function to display images along with bounding boxes and classes is executed. The color coding of the boxes represent whether the object is a vehicle (Red), pedestrian (Blue) or a cyclist (Green).

![images_grid](https://user-images.githubusercontent.com/62600416/205665826-4531011b-4bac-41cd-b005-3ed50d61d2e5.png)

The number of objects in 20000 samples is plotted and compared. It is seen that most of the images has vehicle objects followed by pedestrians and small number of cylists. 

![class_distribution](https://user-images.githubusercontent.com/62600416/205668868-f9460405-ca45-4f24-8ac7-6dd5bcc7fcd4.png)

For the number of objects with count of frames, it is seen that there are lesser images with 0 vehicles and more images with 2-10 vehicles. The distribution decreases with increasing number of vehicles. There are upto 65 vehicles in an image. Is it also seen that there are large number of images with zero pedestrians and some images have upto 45 pedestrians. With the cyclists, it is rare to see image with them, atmost of 5 cyclists are seen in an image.

![frame_and_class_distribution_grid](https://user-images.githubusercontent.com/62600416/205669960-23b024f0-dd4e-4f0d-86c1-543654674d7c.png)

### Cross validation
The dataset is split into training, validation and testing. The validation data is used to check if the model overfits the data. It is not possible to run training and evaluation in parallel. The model is evaluated after its training. The evaluation loss increases when the model overfits

### Training
First the training was performed using the pre-trained model from the zoo of model available on Tensorflow using transfer learning. The model used here is 
SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector ![here](https://arxiv.org/pdf/1512.02325.pdf). The model is loaded to ```reference\pipeline_new.config```. The model is trained for 2500 epochs with folowing code.

First, download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

Edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. Also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. To kill the evaluation script manually using
`CTRL+C`.
The progress of the model's training is done using Tensorboard. Different losses (localization loss, training loss) are monitored along with evaluation metrics recall and precision

#### Reference
#### Reference experiment
The loss curve starts low and starts to oscillate, this oscillation starts to decrease meaning the model is learning. The loss comes to around 0.702 at the end of 2500 epochs. The localization loss and the total loss comes to around 0.75 and 15.31 respectively. This oscillation maybe attributed to high learning rate. 

![reference_loss](https://user-images.githubusercontent.com/62600416/205664903-09f868fd-617c-4260-b8ac-e13267af5244.png)

The evaluation is run for 1 epoch and is around the training loss about 0.759.

The recall and precision from the evaluation metrics reveals that the model is not good enough as all the values are nearly/ close to zero as seen in the image.
![reference_detection_box_precision](https://user-images.githubusercontent.com/62600416/205665215-21c1f29e-06cb-4f82-93ff-941f3f332c83.png)
![reference_detection_box_recall](https://user-images.githubusercontent.com/62600416/205665555-7027cc2c-fdc8-4152-b54e-338c8ca20112.png)

The model is not good. The evaluation metrics support this statement. 
Some tweaks in the config and data could be made to improve the models performance. The improvements made are explained in the next section

#### Improve on the reference
This section highlights the different strategies adopted to improve model.
- From the random surf through the images, it is seen that maximum object occur in the centerm fewer on the sides and lesser on top and bottom of the images.
- There are fewer samples in darker/foggy conditions compared to clear conditions. So, brightness, contrast and color shift augmentations should help to improve the models performance.
- The color of the object is irrelevant, random_rgb_to_gray with a probability of 0.5 is used
- The data class distribution is also uneven, with the vehicle class appearing more and cyclists less. This leads to a bias towards vehicle class.

The output of the data augmentations are visualized using ```Explore data augmentations.ipynb```. A grid of 8 samples is observed as below.

![exp0_img_grid](https://user-images.githubusercontent.com/62600416/205911914-f7a00b3d-b2a7-477e-aacf-c06006ae0ff1.png)

### Edit the config file
The following changes were made to the ```pipline_new.config``` as part of first experiment ```experiment0```.
* The base learning_rate for the cosine decay is decreased from 0.04 to 0.008
* The number of epochs is increased from 2500 to 3000
* The batch size is increased from 2 to 8
* The following data augmentations were added
	* random_rbg_to_gray
	* random_adjust_brightness
	* andom_adjust_contrast
	* random_adjust_saturation

### Experiment0

**Important:** The checkpoints files can be deleted after each experiment. However keep the `tf.events` files located in the `train` and `eval` folder of experiments.  Also keep the `saved_model` folder to create your videos.

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
