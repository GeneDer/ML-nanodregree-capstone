
# This project still under construction!!!


Udacity machine leanring nanodegree capstone project intended to solve the 
detiction problem with number digits in real-world images. Given a real-world
image contains any amount of number digits with any font, color, and size exist 
anywhere in the image. The project is trying to identify the location and the class 
of the digits. It break down the problem into two sub-problems. First, it uses 
state of the art OverFeat for region proporal of digit regions. Second, it uses neural 
network to classify each proposed region for the digit. Special thanks to 
Russell Stewart for open source the implementaion of Overfeat in TensorFlow,
[TensorBox](https://github.com/Russell91/TensorBox).

## Number Detection with pre-trained model
Make sure you have [TensorFlow](https://github.com/tensorflow/tensorflow) installed
on your computer befroe you start.

1. Clone this repository
2. Download `googlenet.pb`, `classification_model.ckpt`, and `overfeat_checkpint.ckpt`
into `data` ditectory via runing `download_models.sh`.
3. Put 


## Download Dataset

