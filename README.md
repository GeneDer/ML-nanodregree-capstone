
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


## Number Detection With Pre-trained Model
Make sure you have [TensorFlow](https://github.com/tensorflow/tensorflow) installed
on your computer befroe you start. You may also need to install `numpy`, `scipy`, 
`PIL`, and verious of python libraries if you don't already have them.

1. Clone this repository.
2. Download `googlenet.pb`, `classification_model.ckpt`, and `overfeat_checkpint.ckpt`
into `data` ditectory via runing `download_models.sh`.
3. Put the your image(s) in `images_input` folder.
4. If you only have one image to evaluate, enter `python evaluation.py <image.jpg>`
in Terminal. Make sure to replace <image.jpg> with the file name of your image and
to include the filename extension.
5. If you want to evaluate on all images in the folder, simply enter `python evaluation.py`
in Terminal.
6. The Terminal will show the average time used for each image evaluation once all the 
images are evaluated. The results will be in the `images_output` folder. A json with 
all detected digits will be generated in `data` folder.


## Train on New Data
Here I will go through the steps to train on the [SVHN](http://ufldl.stanford.edu/housenumbers/)
dateset. You can replace the data with your own dataset.

### Download and Preprocess Datasets
1. Download and extract [train.tar.gz](http://ufldl.stanford.edu/housenumbers/train.tar.gz), 
and [test.tar.gz](http://ufldl.stanford.edu/housenumbers/test.tar.gz) into the root
of this repository.
2. If you have Matlab installed on your computer, you can dump `mat_to_txt.m` into 
`train` and `test` directory to convert `digitStruct.mat` to txt file.
3. You can also get my generated verson from [train](https://www.dropbox.com/s/jmmb9jzaiqr9dhp/train.txt?dl=1)
and [test](https://www.dropbox.com/s/8394po4yqmbi2s6/test.txt?dl=1). Dump each txt into their corresponding
folder. If you are interested in training on bigger set, I also have 
[extra](https://www.dropbox.com/s/kx600daed60v2no/extra.txt?dl=1) available.
4. 


