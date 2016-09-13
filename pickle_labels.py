from six.moves import cPickle as pickle
import numpy as np
import os


def load_labels(folder):
  print folder
  maximum_size = 100000
  image_files = os.listdir(folder)
  labels = np.ndarray(shape=(min(len(image_files), maximum_size), 1),
                      dtype=np.int32)
  numbers = {}
  count_line = 0
  with open(folder+'/'+folder+'.txt', 'r') as f:
    for line in f:
      parse_line = line.split(':')
      number = ''
      for i in xrange(1,len(parse_line)):
        n = parse_line[i].split(',')[0]
        if n != '10':
          number += n
        else:
          number += '0'
      labels[count_line] = int(number)
      count_line += 1
      if count_line >= maximum_size:
        break
      
  labels = labels[0:count_line, :]
  print('Full dataset tensor:', labels.shape)
  return labels

        
def pickle_labels(folder):
  dataset_names = []
  set_filename = folder + '_labels.pickle'
  
  if os.path.exists(set_filename):
    print('%s already present - Skipping pickling.' % set_filename)
  else:
    dataset = load_labels(folder)
    with open(set_filename, 'wb') as f:
      pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    
  return dataset_names


train_labels = pickle_labels('train')
test_labels = pickle_labels('test')
extra_labels = pickle_labels('extra')
