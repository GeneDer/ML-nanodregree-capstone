from six.moves import cPickle as pickle
import numpy as np
import os
from scipy import ndimage, misc

"""
#import matplotlib.pyplot as plt

pixel_depth = 255.0

train_folder = 'train'
image_data1 = (ndimage.imread('train/1.png', flatten = True).astype(float) - 
              pixel_depth / 2) / pixel_depth
image_data2 = (ndimage.imread('train/2.png', flatten = True).astype(float) - 
              pixel_depth / 2) / pixel_depth
print image_data1.shape
print image_data2.shape
resize1 = misc.imresize(image_data1, (100,100))
resize2 = misc.imresize(image_data2, (100,100))
print resize1.shape
print resize2.shape
#plt.imshow(resize1)
#plt.show()
#plt.imshow(resize2)
#plt.show()

i1 = ndimage.imread('MDEtMDEtMDAudHRm.png')
image_data = (ndimage.imread('MDEtMDEtMDAudHRm.png').astype(float) - 
              pixel_depth / 2) / pixel_depth
print i1
print image_data
print image_data.shape
resize3 = (misc.imresize(image_data, (100,100))- 
              pixel_depth / 2) / pixel_depth

print resize3.shape
#plt.imshow(resize3)
#plt.show()

dataset = np.ndarray(shape=(1, 100, 100),
                         dtype=np.float32)
dataset[0, :, :] = resize3
print('Full dataset tensor:', dataset.shape)
print('Mean:', np.mean(dataset))
print('Standard deviation:', np.std(dataset))
"""


image_size = 64  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_image(folder):
  print folder
  maximum_size = 100000
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(min(len(image_files), maximum_size), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  for i in xrange(1,len(os.listdir(folder))):
    image_file = os.path.join(folder, str(i)+'.png')
    try:
      image_data = ndimage.imread(image_file, flatten = True)
      image_data_resized = misc.imresize(image_data, (image_size, image_size))
      image_data_normalized = (image_data_resized.astype(float) -
                               pixel_depth / 2) / pixel_depth
 
      dataset[image_index, :, :] = image_data_normalized
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    if image_index >= maximum_size:
      break
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def pickle_images(folder):
  dataset_names = []
  set_filename = folder + '.pickle'
  
  if os.path.exists(set_filename):
    print('%s already present - Skipping pickling.' % set_filename)
  else:
    dataset = load_image(folder)
    with open(set_filename, 'wb') as f:
      pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    
  return dataset_names

test_datasets = pickle_images('test')
train_datasets = pickle_images('train')
extra_datasets = pickle_images('extra')
