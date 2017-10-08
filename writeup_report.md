**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/offroad.jpg "Recovery Image"
[image4]: ./examples/offroad2.jpg "Recovery Image"
[image5]: ./examples/offroad3.jpg "Recovery Image"
[image2]: ./examples/normal.jpg "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a fully convolution neural network with 4 layers with varying kernel sizes and strides. After doing a few experiments it semed to me that having some dense layers following the convolutional ones didn't add to the accuracy of the network, at least whith the dataset I used.

The data is normalized in the model using a Keras lambda layer performing the operation x/126+1, turning the RGB map ints (0-255) to floats ranging from -1 to 1. 

The images are also cropped at the top.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was validated using a different validation set.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I created several dataset: the normal dataset (one for track 1 and one for track 2) showing some full laps, a curve dataset about the steeper curves in the first track and the shadows in the second, and lastrly a recovery dataset from the sides/off-road

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy was to find a network that gave an acceptable output and try to reduce its size without worsening the performance.

My first step was to use a convolution neural network model similar to alexnet, but with a single output. I thought this model might be appropriate as a starting point because it is light and easy to adapt.

Doing some experiments I noticed that the last fully connected layers didn't seem to help much. I tried rising the size and stride of the first kernels, thinking that a large kernel would better capture information from such a large image, and it seemed to pay off.

I used dropout layers to reduce the overfitting, experimenting with various dropout rates.

I used the autonomous driving as a benchmark to see how the network really fared, as low validation error didn't always correspond to better performance.

The first track's main issues were the curves and the bridge. In particular, while the curves could be handled with a better dataset, adding a dataset specifically for the bridge lowered the performance of the network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track one, it does well on most of track2, but I was unable to handle the last problematic curve (steep and with odd lighting) so far.

#### 4. Architecture

The model is a fully convolutional layer, with a single neuron output, as follows:

| kernel size | depth | strides | activation function|
|:-----------:|:-----:|:-------:|:------------------:|
|15*15        |24     |3*6      |relu                |
|7*7          |48     |3*3      |relu                |
|5*5          |96     |         |relu                |
|3*3          |192    |         |relu                |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer towards the center if it finds itself out of the ideal path: 

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles in order to remove any steering bias and to generalize the learning process


I finally randomly shuffled the data set and put 2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs on my dataset was roughly 60 as evidenced by the more coherent driving on track1 and the better result on track2, compared to fewer or more epochs.

It seems the network is only able to handle the shadowy part of track2 after a certain number of epochs, while it more or less handles the rest of the tracks from the first epochs, I believe the reason is that the lower value of the input scales down the output to a level it is very close to zero.


I used an adam optimizer so that manually training the learning rate wasn't necessary.
