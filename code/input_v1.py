from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import random

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from PIL import Image

#data_dir = '/Users/phuongpham/Documents/CanePhotoToKmut'
data_dir = '/Users/phuongpham/Documents/cane'

def photo_records_to_txt():
  '''
  Create a .txt file with one  /path/to/image label per line 
  '''
  #WD = r'/Users/phuongpham/Documents/CanePhotoToKmut/7-9months'
  WD = r'/Users/phuongpham/Documents/cane'
  png_files_path = glob.glob(os.path.join(WD, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(WD, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(WD, '*.[jJ][pP][gG]'))

  #files = glob.glob(os.path.join(WD, '*.jpg'))
  label = '0' 
  with open('/Users/phuongpham/Documents/infiles.txt', 'w') as in_files:
    if jpg_files_path:
      for fn in jpg_files_path:
        if fn.find('good') != -1:
          label = '2'
        if fn.find('medium') != -1:
          label = '1'
        if fn.find('poor') != -1:
          label = '0'
        in_files.writelines(fn + ' '+label+' '+'\n')
    if jpeg_files_path:
      for fn in jpeg_files_path:
        if fn.find('good') != -1:
          label = '2'
        if fn.find('medium') != -1:
          label = '1'
        if fn.find('poor') != -1:
          label = '0'
        in_files.writelines(fn + ' '+label+' '+'\n')
    if png_files_path:
      for fn in png_files_path:
        if fn.find('good') != -1:
          label = '2'
        if fn.find('medium') != -1:
          label = '1'
        if fn.find('poor') != -1:
          label = '0'
        in_files.writelines(fn + ' '+label+' '+'\n')
        
def dense_to_one_hot(labels_dense, num_classes=3):
  '''
  Convert class labels from scalars to one-hot vectors
  0 => [1 0 0 0 0 0 0 0 0 0]
  1 => [0 1 0 0 0 0 0 0 0 0]
  ...
  '''
  
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(one_hot,num_classes):
  '''
  Reads a .txt file containing pathes and labeles
  Args:
     image_list_file: a .txt file with one /path/to/image per line
     label: optionally, if set label will be pasted after each line
  Returns:
     List with all filenames in file image_list_file
  '''
  
  photo_records_to_txt()
  image_list_file = '/Users/phuongpham/Documents/infiles.txt'
  f = open(image_list_file, 'r')
  filenames = []
  labels = []
  for line in f:
    #filename, label = line[:-1].split(' ')
    filename, label = line.strip().split(' ', 1)
    filenames.append(filename)
    labels.append(int(label))
  if one_hot:
    labels = np.array(labels)
    labels = dense_to_one_hot(labels, num_classes)
    print (labels)
  return filenames, labels

def distorted_image(image):
  '''
  Randomly distord the seed image to increase the training size as well as to generalize the dataset  
  '''
  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(image)
  #good value = 0.1, 0.05
  distorted_image = tf.image.adjust_brightness(distorted_image,delta=random.uniform(0.05,0.1)) 
  #good value = 1.1 - 1.3 
  distorted_image = tf.image.adjust_contrast(distorted_image,contrast_factor = random.uniform(1.1,1.3))
  #good value = 0.7-0.9 
  distorted_image = tf.image.central_crop(distorted_image,central_fraction=random.uniform(0.7,0.9))
  resized = tf.image.resize_images(distorted_image, 500, 500, 1)
  #resized.set_shape([500,500,3])
  
  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
  #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)  
  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_whitening(distorted_image)
  return resized

def read_image(filename_queue):
  #Arg: a list of filename
  #Return: a tensor of resized image 
  
  # Read an entire image file which is required since they're JPEGs, if the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  reader = tf.WholeFileReader()
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  key,value = reader.read(filename_queue)
  # Decode the image as a JPEG file, this will turn it into a Tensor which we can
  # then use in training.
  image = tf.image.decode_jpeg(value)
  return key,image
  
def inputs():
  '''
  Return a tensor of distorded image 
  '''
  
  filenames = []
  labels = []
  filenames, labels = extract_labels(one_hot=True, num_classes=3)
  #filenames = tf.train.match_filenames_once("/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/*.jpg")
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good3.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good4.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good5.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium3.jpeg']
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg']
  filename_queue = tf.train.string_input_producer(filenames,num_epochs=5)
  filename,read_input = read_image(filename_queue)
  reshaped_image = distorted_image(read_input)
  filenames, labels = extract_labels(one_hot=True, num_classes=3)
  filename,read_input = read_image(filename_queue)
  return filename,reshaped_image


with tf.Graph().as_default():
  #image = extract_labels(one_hot = True, num_classes = 3)
  image = inputs()
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  '''
  for i in xrange(5):
    filename,img = sess.run(image)
    img = Image.fromarray(img, "RGB")
    img.save(os.path.join(data_dir,"foo"+str(i)+".jpg"))
  '''
  for i in xrange(5):
    filename,img = sess.run(image)
    img = Image.fromarray(img, "RGB")
    if filename.find('good') != -1:
      img.save(os.path.join(data_dir,"good"+str(i)+".jpg"))
    if filename.find('medium') != -1:
      img.save(os.path.join(data_dir,"medium"+str(i)+".jpg"))
    if filename.find('poor') != -1:
      img.save(os.path.join(data_dir,"poor"+str(i)+".jpg"))
      

"""
filenames = ['/Users/phuongpham/Documents/untitled folder/CanePhoto_toKmut/7-9months/good2.jpg']
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key,value = reader.read(filename_queue)
image = tf.image.decode_jpeg(value,channels=3)
print(image.get_shape())

resized = tf.image.resize_images(image, 180,180, 1)
resized.set_shape([180,180,3])
print(resized.get_shape())
  
init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
with sess.as_default():
  sess.run(init_op)
  tf.train.start_queue_runners(sess=sess)
    for i in xrange(2):
  img = sess.run(image)
  img = Image.fromarray(img, "RGB")
  img.save(os.path.join(data_dir,"foo"+str(i)+".jpg"))
  
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(1): #length of your filename list
  image = resized.eval(session=sess) #here is your image Tensor :) 

print(image.shape)
Image.fromarray(np.asarray(image)).show()

coord.request_stop()
coord.join(threads)
"""
"""
def read_raw_images(path, is_directory=True):
  Reads directory of images in tensorflow
  Args:
    path:
    is_directory:
  Returns:
 
  images = []
  png_files = []
  jpeg_files = []

  reader = tf.WholeFileReader()

  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))

  if is_directory:
    for filename in png_files_path:
      png_files.append(filename)
    for filename in jpeg_files_path:
      jpeg_files.append(filename)
    for filename in jpg_files_path:
      jpeg_files.append(filename)
  else:
    raise ValueError('Currently only batch read from directory supported')

  # Decode if there is a PNG file:
  if len(png_files) > 0:
    png_file_queue = tf.train.string_input_producer(png_files)
    pkey, pvalue = reader.read(png_file_queue)
    p_img = tf.image.decode_png(pvalue)
    reshaped_image = modify_image(p_img)
    reshaped_image = tf.cast(reshaped_image, tf.float32)

  if len(jpeg_files) > 0:
    jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)
    reshaped_image = modify_image(p_img)
    reshaped_image = tf.cast(p_img, tf.float32)

  return 

init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
with sess.as_default():
    sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(1): #length of your filename list
  image = my_img.eval() #here is your image Tensor :) 

Image.fromarray(np.asarray(image)).show()

coord.request_stop()
coord.join(threads)"""
'''
def photo_records_to_txt():
  
    Create a .txt file with one  /path/to/image label per line 
  
  filename = '/Users/phuongpham/Documents/CanePhoto_toKmut/7-9months/good2.jpg'
  label = 0
  if filename.find('good') != -1:
    label = 2
  else if filename.find('medium') != -1:
    label = 1
  label = int(label)
  return label

def read_image(filename_queue):
  reader = tf.WholeFileReader()
  key,value = reader.read(filename_queue)
  image = tf.image.decode_jpeg(value,channels=3)
  resized = tf.image.resize_images(image, 180, 180, 1)
  resized.set_shape([180,180,3])
  return key,image

'''  