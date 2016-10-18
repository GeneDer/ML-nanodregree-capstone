from six.moves import cPickle as pickle
import numpy as np
import os
from scipy import ndimage, misc


image_size = 28
pixel_depth = 255

def process_data(folder):
  print folder
  image_array = []
  label_array = []
  count_line = 0
  with open("%s/%s.txt"%(folder, folder), 'r') as f:
    for line in f:
      parse_line = line.strip().split(':')
      
      image_data = ndimage.imread('%s/%s'%(folder, parse_line[0]), flatten = True)

      for i in xrange(1, len(parse_line)):
        element_split = map(int, parse_line[i].split(','))
        x1 = element_split[2]
        y1 = element_split[1]
        x2 = x1 + element_split[4] + 1
        y2 = y1 + element_split[3] + 1
        if x1 < 0:
          x1 = 0
        if y1 < 0:
          y1 = 0
        try:
          number = misc.imresize(image_data[y1:y2, x1:x2], (image_size, image_size))
          image_array.append((number.astype(float) - pixel_depth / 2) / pixel_depth)
          label_array.append(element_split[0])
        except:
          print image_data.shape, element_split

  image_array = np.array(image_array)
  label_array = np.array(label_array)
  print('Data length:', image_array.shape)
  print('Mean:', np.mean(image_array))
  print('Standard deviation:', np.std(image_array))
  return image_array, label_array

def pickle_data(folder):
  dataset_names = []
  set_filename = folder + '_clf.pickle'
  
  if os.path.exists(set_filename):
    print('%s already present - Skipping pickling.' % set_filename)
  else:
    dataset = process_data(folder)
    with open(set_filename, 'wb') as f:
      pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    
  return dataset_names
        

test_datasets = pickle_data('test')
train_datasets = pickle_data('train')
