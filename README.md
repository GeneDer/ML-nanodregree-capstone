
# Detect and Classify Numbers in Real-World Images

This prject is written for Udacity machine leanring nanodegree capstone project 
that intended to solve the detiction problem with number digits in real-world 
images. Given a real-world image that contains any amount of number digits with any 
font, color, and/or size exist anywhere in the image. The project is trying 
to identify the location and the class of each of the digits. It breaks down 
the problem into two sub-problems. First, it uses state of the art OverFeat 
for region proporal of digit regions. Second, it uses neural 
network to classify each proposed region for the digit. Special thanks to 
Russell Stewart for open source the implementaion of Overfeat in TensorFlow,
[TensorBox](https://github.com/Russell91/TensorBox).

![Alt text](images_output/00.png?raw=true "Optional Title")


## Number Detection With Pre-trained Model
Make sure you have [TensorFlow](https://github.com/tensorflow/tensorflow) installed
on your computer befroe you start. You may also need to install `numpy`, `scipy`, 
`PIL`, and verious of python libraries if you don't already have them.

1. Clone this repository.
2. Download `googlenet.pb`, `classification_model.ckpt`, and `overfeat_checkpint.ckpt`
into `data` directory via runing `download_models.sh`.
3. Put the your image(s) in `images_input` folder.
4. If you only have one image to evaluate, enter `python evaluation.py <image.jpg>`
in Terminal. Make sure to replace <image.jpg> with the file name of your image and
to include the filename extension.
5. If you want to evaluate on all images in the folder, simply enter `python evaluation.py`
in Terminal.
6. There are already 21 images in the `images_input` folder that you can try it 
out yourself.
7. The Terminal will show the average time used for each image evaluation once all the 
images are evaluated. The results will be in the `images_output` folder. A .json file 
with all detected digits will also be generated in `data` folder.


## Training on New Data
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
4. Run `overfeat_data_processing.py` in `SVHN` folder. It will automaticlly generate
resized images, .idl, and .json for [TensorBox](https://github.com/Russell91/TensorBox)
to train on.
5. Copy `overfeat_rezoom.json` from the root of this repository into TensorBox's 
`hypes` folder and copy `SVHN` folder from this repository into TensorBox's 
`data` folder. You should be ready to train with TensorBox.
6. Run `character_classification_data_processing.py` in this repository, it should 
automaticlly generate the pickle files of image data and labels for our 
classification networks.

### Training and Perpare for Evaluation
1. Region proporal will be trained using [TensorBox](https://github.com/Russell91/TensorBox).
Please refer to TensorBox for more detail. 
2. After TensorBox is trained, rename the .ckpt file with `overfeat_checkpint.ckpt`
and dump it into `data` folder.
3. Run `classification_networks.py` to train the classification networks.
4. The check point file will be stored in `/tmp` folder. Rename the .ckpt file
with `classification_model.ckpt` and dump it into `data` folder.
5. If you don't alreayd have `googlenet.pb` in the `data` folder, download it from
[here](http://russellsstewart.com/s/tensorbox/googlenet.pb).
6. After all files are trained and in place, run `python evaluation.py` to evluate
all images in `images_input` folder or `python evaluation.py <image.jpg>` for a 
specific image. Results will be shown in `images_output` folder.