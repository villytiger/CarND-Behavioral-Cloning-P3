# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/img1.png "Model Visualization"
[image2]: ./img/img2.jpg "Center lane driving"
[image3]: ./img/img3.jpg "Recovery Image"
[image4]: ./img/img4.jpg "Recovery Image"
[image5]: ./img/img5.jpg "Recovery Image"
[image6]: ./img/img6.jpg "Normal Image"
[image7]: ./img/img7.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py and model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 containing a video recording of my vehicle driving autonomously 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is implemented using Nvidia architecture of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 78-83) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using appropriate Keras layers (code lines 75-76). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 87, 89, 91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in a backward direction and center lane driving on tack 2 in both directions.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

From the beginning I used Nvidia architecture. It fits well for this work. The main problem was collecting proper training data.

I found that my model stops learning after a few epochs. So the best way to avoid overfitting was running train process for 2-3 epochs. Dropout layers in this project could help against overfitting but just a little.

At first I collected data on 3 laps in both directions from track 1. With initial model I could get good performance on training and validation sets. But when I run my model on simulator it drove randomly ignoring lane lines.

Then I tried using precollected data from Udacity. My model could drive around the track, but sometimes it prefered to run on a roadside.

I tried adding data from recovering laps. The purpose of this data is to teach model to avoid crossing or even coming close to a roadside. I also added data from track 2, but I think it doesn't help much for driving on track 1.

Adding data from left and right cameras with 0.2 correction for angle also helped to teach keeping in the center of the road.

With all that data my model was able to drive until the end of the bridge. After the bridge there is a difficult part of the track where roadside looks different. My model drove straight ignoring turn of the road. For eliminating this bug I used larger angle in recovering data on this place.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 73-95) consisted of a convolution neural network with the following layers and layer sizes:
1. convolution 5x5 with 24 features
2. ELU
3. max pooling 2x2
4. convolution 5x5 with 36 features
5. ELU
6. max pooling 2x2
7. convolution 5x5 with 48 features
8. ELU
9. max pooling 2x2
10. convolution 3x3 with 64 features
11. ELU
12. convolution 3x3 with 64 features
13. ELU
14. hidden layer with 100 units
15. ELU
16. hidden layer with 50 units
17. ELU
18. hidden layer with 10 units
19. ELU
20. output linear layer with 1 unit

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to driving back to center of road from roadside. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would augment training data with minimal costs. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 54248 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by evaluation on the validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
