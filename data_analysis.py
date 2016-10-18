from scipy import ndimage, misc

def process_data(folder):
  print folder
  count_line = 0
  digit_count = {}
  total_image_width = 0
  total_image_height = 0
  totl_digit_width = 0
  total_digit_height = 0
  class_count = [0]*11
  
  with open("%s/%s.txt"%(folder, folder), 'r') as f:
    for line in f:
      count_line += 1
      
      parse_line = line.strip().split(':')
      if len(parse_line) - 1 in digit_count:
        digit_count[len(parse_line) - 1] += 1
      else:
        digit_count[len(parse_line) - 1] = 1
      
      image_data = ndimage.imread('%s/%s'%(folder, parse_line[0]))
      image_height, image_width, image_depth = image_data.shape
      total_image_width += image_width
      total_image_height += image_height
      
      
      for i in xrange(1, len(parse_line)):
        element_split = map(int, parse_line[i].split(','))
        totl_digit_width += element_split[4]
        total_digit_height += element_split[3]
        class_count[element_split[0]] += 1

  total_digits_count = 0
  for i in digit_count:
    total_digits_count += i*digit_count[i]
  mean_digit_count = float(total_digits_count)/count_line
  SD_digit_count = 0
  for i in digit_count:
    SD_digit_count += (i - mean_digit_count)**2
  SD_digit_count /= count_line
  SD_digit_count = SD_digit_count**0.5
  
  print 'Total image count:', count_line
  print 'Total digits count:', total_digits_count
  print 'Average digits per image:', mean_digit_count
  print 'SD digits per image:', SD_digit_count
  print 'Average image width:', float(total_image_width)/count_line
  print 'Average image height:', float(total_image_height)/count_line
  print 'Average digit width:', float(totl_digit_width)/count_line
  print 'Average digit height:', float(total_digit_height)/count_line
  print 'Number of digits count:', digit_count
  print 'Classes Count:', class_count

process_data("test")
process_data("train")
