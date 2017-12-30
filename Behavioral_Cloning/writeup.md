# **Behavioral Cloning**

## Intro
Behavioral cloning is a process of learning by imitation in which, human skills are captured and reproduced by a computer program. Behavioral cloning is generally used in tasks where the set of rules are too many to hard code using conditional statements. The steps in Behavioral cloning include

- recording data of human performing a skill
- training a machine learning model to learn the set of rules and nuances that underlie the skill
- reproducing the skill on a similar problem in different environment

In this project, the task is to
- record data of a human driving a car around a track
- choose and train a suitable machine learning model to learn the driving skill
- let the computer drive the car autonomously on the track


[//]: # (Image References)

[image1]: images/sample_img.jpg "Sample Training Image"
[image2]: images/sample_img_cropped.jpg "Sample Cropped Training Image"
[image3]: images/lenet_architecture.png "Modified LeNet-5 Architecture"
[image4]: images/comma_architecture.png "Comma.ai Architecture"
[image5]: images/nvidia_architecture.png "Nvidia CNN Architecture"
[image6]: images/center_img.jpg "Track 1 Center Lane Driving"
[image7]: images/recovery1.gif "Recovering From Edges"
[image8]: images/recovery2.gif "Recovering From Edges"
[image9]: images/center_img2.jpg "Track 2 Center Lane Driving"
[image10]: images/sample_img2.jpg "Sample Image"
[image11]: images/sample_img2_flipped.jpg "Sample Image Flipped"
[image12]: images/jerky.gif "Keyboard Input"
[image13]: images/smooth.gif "Controller Input"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
- `train.py` : script to read and augment images in dataset
- `models.py` : contains different model definitions in Keras with TensorFlow backend
- `drive.py` : script for driving car in autonomous mode
- `models/model_lenet.h5` : model trained with modified LeNet-5 based architecture
- `models/model_comma.h5` : model trained with Comma.ai architecture
- `models/model_lenet.h5` : model trained with Nvidia architecture
- `writeup.md` : writeup summarizing the results

#### 2. Submission includes functional code

To train a model on your own dataset use the below command, and the script outputs `model_<model_name>.h5` file in the same directory as `train.py`

```
python train.py <dataset path> <lenet|comma|nvidia>

Ex: python train.py datasets/track_1_2/ nvidia
```

The trained model can be used to drive the car in Udacity simulator using the below command:

```
python drive.py models/model_nvidia.h5
```

To record the images of the car driving in autonomous mode pass the directory name as the 2nd argument:

```
python drive.py models/model_nvidia.h5 <output image folder>

Ex: python drive.py models/model_nvidia.h5 output
```

#### 3. Submission code is usable and readable

The `train.py` and `models.py` files contain the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### 4. Has an appropriate model architecture been employed for the task?

I have tried 3 different models: LeNet-5, Comma.ai, Nvidia model, all the models have normalization and activation layer (ELU in LeNet-5 and Comma.ai models, ReLU in Nvidia model).

#### 5. Has an attempt been made to reduce overfitting of the model?

I didn't experience the problem of low loss in training set and high loss on validation set. In all the cases, the validation loss was always lower than training loss. So, I didn't add Dropout layer to my final model (Nvidia). To reduce over-fitting, I tried to generalize the model by recording data in both clockwise and anti-clockwise direction, and my training set also included images from the second track.

#### 6. Have the model parameters been tuned appropriately?

I used Adam Optimizer to train my model so that manually training the learning rate wasn't necessary.

#### 7. Is the training data chosen appropriately?

My training data includes 2 laps of center lane driving in clockwise and anti-clockwise direction on track 1 and track 2. It also includes data for recovering from driving close to edges and recovering from driving away from the center lane. In addition to that, it includes additional data for driving smoothly around the curves. More information about the training data and process can be found in [Creation of Training Set and Training Process](#3-creation-of-training-set-and-training-process) section.

#### 8. Is the solution design documented?

The README file is same as the writeup (this file), except for the rubrics part. It thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

#### 9. Is the model architecture documented?

I have described the 3 network architectures used in this project: LeNet-5, Comma.ai, Nvidia, and I have also included visualization of network architecture for each of the model in [Solution Design Approach](#1-solution-design-approach) and [Final Network Architecture](#2-final-network-architecture) sections. In addition to visualization of layers, I have also reasoned the use of normalization and cropping layer along with images.

#### 10. Is the creation of the training dataset and training process documented?

I have documented the training process in the [Creation of Training Set and Training Process](#3-creation-of-training-set-and-training-process) section

#### 11. Is the car able to navigate correctly on test data?

Here is the link to video of the car driving in autonomous mode on [track 1](videos/track_1_autonomous.mp4) and [track 2](videos/track_2_autonomous.mp4).


## Network Architecture and Training Strategy

### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to choose existing powerful network architectures and modify it to predict steering angle based on input image.

My initial approach was to use [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) architecture. This architecture is designed for recognizing handwritten digits and it takes 32x32x1 image as input and outputs classification score for 10 digits. The use-case in this project is different, instead of classification of images, the model has to predict steering angle from input images,  and the input images are larger, 160x320x3 pixels. To suite the needs, I modified the LeNet-5 architecture to read larger images and output steering angle predictions.

I noticed that the model trained very slowly and required around 10 epochs to get acceptable performance. To combat this problem, I pre-processed the data (normalize and mean-center) by adding lambda layer at the beginning of the network. Pre-processing using lambda layer helps to parallelize the pre-processing step over multiple GPU threads and reduces training time. In addition to reducing the training time, lambda layer will also normalize and mean-center input images when making steering predictions. After adding data pre-processing step the training time reduced to 4 to 5 epochs.

Pre-processing the data only improved the training performance, but the car didn't drive smoothly on the track, it wavered a lot about the center of the track and veered off the track in many cases. I noticed that the training images contained lot of unnecessary information like hood of the car, hills, trees and other landscape, so I added cropping layer to my model to crop the segment of the image which contains only the road.

![alt text][image1]
![alt text][image2]

The result of the above modifications were decent, the car mostly stayed on the track on straight roads, but it still had difficulty in staying on the track in curves.

![alt text][image3]

Next, I tried the model described in [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) github page. This model consists of 7 layers which includes, 1 normalization layer followed by 3 convolution layers, a dropout layer, fully connected layer and finally another dropout layer. This model performed similar to the LeNet-5 based model.

![alt text][image4]

Finally, I implemented the network architecture described in [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) research paper published by Autonomous Vehicle team at Nvidia. I modified the architecture by adding the cropping layer after normalization so that the network can train only on the useful information in the training images. This network worked well and the car stayed on center of the center track all the time, even in the curves, and the car drove smoother than the LeNet-5 based model. Overall, the Nvidia model produced the best results.

## 2. Final Network architecture
The final network architecture (models.py : lines 36 to 62) consisted 10 layers: a normalization layer, cropping layer followed by 5 convolution layers and 3 fully connected layers. The figure below shows the different layers of the architecture and the input and output shapes.

**Note:** the Lambda layer corresponds to pre-processing(normalize and mean-center) of input images

![alt text][image5]

## 3. Creation of Training Set and Training Process
To capture good driving behavior, I recorded images of driving in the center of the road for two laps of track one. To generalize the model, I also recorded two laps of driving in the counter-clockwise direction. Here is an example image of center lane driving:

![alt text][image6]
![alt text][image9]

One of the main reason for the car veering off the track was that the model did not know what to do when the car moves closer to edges. To resolve this problem, I recorded the car recovering from the left and right edge of the track so that the would learn to recover from driving off the center lane. The images below show what a recovery looks like:

![alt text][image7]
![alt text][image8]

I repeated the process on track two to get more data points.

After collecting data from both the tracks, I flipped the images and corresponding steering angle so that it would look like the car is driving in the opposite direction and add mode data to generalize the model. Here is an example that has been flipped:

![alt text][image10]
![alt text][image11]

I also noticed that the images read using OpenCV are in BGR colorspace, but the images read in drive.py using PIL were in RGB colorspace. I converted the color space of images read using OpenCV to RGB colorspace *(train.py : line 24)* to have uniform colorspace for images during training and prediction.

One of the important step in getting smooth driving performance was to record the training data using controller (I used Sony DualShock 4) instead of keyboard. The advantage of using a controller is that it has more granular control of steering, so it's easy to maintain constant steering angle in curves. This enables the model to learn to drive the car smoothly around the curves.

![alt text][image12]
![alt text][image13]

Finally, I randomly shuffled the data set and used SciKit-Learn's `train_test_split` function to split the dataset into 70% for training and 30%
for validation. The validation set helped me to determine the performance of my model in each epochs. Initially I had set the number of training epochs to 10, but I noticed that the loss decreased only for first 4 epochs and then increased and decreased randomly for the remaining epochs. So I reduced the number of training epochs to 4. I used Adam Optimizer to train my model so that manually training the learning rate wasn't necessary.
