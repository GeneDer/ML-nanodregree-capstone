from six.moves import cPickle as pickle
import numpy as np
import os


def load_labels(folder):
  print folder
  maximum_size = 100000
  image_files = os.listdir(folder)
  labels = {'length':np.ndarray(shape=(min(len(image_files), maximum_size), 1),
                                dtype=np.int32),
            'bbox':[], 'number':[]}

  count_line = 0
  with open(folder+'/'+folder+'.txt', 'r') as f:
    for line in f:
      parse_line = line.replace('\n', '').split(':')
      number = []
      bbox = []
      for i in xrange(1, len(parse_line)):
        element_split = map(int, parse_line[i].split(','))
        if element_split[0] != 10:
          number.append(element_split[0])
        else:
          number.append(0)
        bbox.append(element_split[1:5])
                        
      labels['length'][count_line] = len(parse_line) - 1
      labels['bbox'].append(bbox)
      labels['number'].append(number)
      
      count_line += 1
      if count_line >= maximum_size:
        break
      
  labels['length'] = labels['length'][0:count_line, :]
  print 'length shape:', labels['length'].shape
  print 'bbox shape:', len(labels['bbox'])
  print 'number shape:', len(labels['number'])
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
