from six.moves import cPickle as pickle
import numpy as np
import os
from scipy import ndimage, misc
import json

def process_data(folder):
  print folder
  json_output = []
  count_line = 0
  with open("../%s/%s.txt"%(folder, folder), 'r') as f, open("%s.idl"%folder, 'w') as o:
    for line in f:
      parse_line = line.strip().split(':')
      tmp_dict = {"image_path": "%s/%s"%(folder, parse_line[0]),
                  "rects": []}      
      o.write("\"%s\":"%(tmp_dict["image_path"]))
      
      image_data = ndimage.imread('../%s'%(tmp_dict["image_path"]))
      image_height, image_width, image_depth = image_data.shape
      misc.imsave('%s'%(tmp_dict["image_path"]),
                  misc.imresize(image_data, (480, 640)))

      for i in xrange(1, len(parse_line)):
        element_split = map(float, parse_line[i].split(','))
        x1 = int(element_split[2]/image_width*640)
        y1 = int(element_split[1]/image_height*480)
        x2 = x1 + int(element_split[4]/image_width*640)
        y2 = y1 + int(element_split[3]/image_height*480)
        if i > 1:
          o.write(', (%s.0, %s.0, %s.0, %s.0)'%(x1, y1, x2, y2))
        else:
          o.write(' (%s.0, %s.0, %s.0, %s.0)'%(x1, y1, x2, y2))
        tmp_dict["rects"].append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2})

      count_line += 1
      json_output.append(tmp_dict)
      if ((folder == "train" and count_line == 33402) or
          (folder == "test" and count_line == 13068)):
        o.write(".\n")
      else:
        o.write(";\n")
  
  json.dump(json_output, open('%s.json'%folder, 'w'))
  
  print('Count line:', count_line)
  print('Json length:', len(json_output))
  return count_line
        

test_datasets = process_data('test')
train_datasets = process_data('train')
