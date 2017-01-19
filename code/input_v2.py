#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import random
import math 
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from PIL import Image

#data_dir = '/Users/phuongpham/Documents/CanePhotoToKmut'
#data_dir = '/Users/phuongpham/Documents/cane'
data_dir_train = '/Users/phuongpham/Documents/cane/train'
data_dir_eval = '/Users/phuongpham/Documents/cane/eval'
txt_dir_train = '/Users/phuongpham/Documents/infiles_train.txt'
txt_dir_eval = '/Users/phuongpham/Documents/infiles_eval.txt'
# Convolutional Layer 1.
filter_size1 = 6          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 5 x 5 pixels.
num_filters3 = 128         # There are 36 of these filters.

# Convolutional Layer 4.
filter_size4 = 3          # Convolution filters are 5 x 5 pixels.
num_filters4 = 128         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 2048             # Number of neurons in fully-connected layer.
img_size = 45
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 3
num_classes = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]



"""Count number of images in train/eval directory"""
def file_len(txt_dir):
  with open(txt_dir) as f:
     len = sum(1 for _ in f)
     print (len)
  return len

def photo_records_to_txt(image_dir,txt_dir):
  '''
  Create a .txt file with one  /path/to/image label per line 
  '''
  #WD = r'/Users/phuongpham/Documents/CanePhotoToKmut/7-9months'
  #WD = r'/Users/phuongpham/Documents/cane'
  png_files_path = glob.glob(os.path.join(image_dir, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(image_dir, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(image_dir, '*.[jJ][pP][gG]'))

  #files = glob.glob(os.path.join(WD, '*.jpg'))
  label = '0' 
  with open(txt_dir, 'w') as in_files:
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

def extract_labels(image_dir,txt_dir,one_hot,num_classes):
  '''
  Reads a .txt file containing pathes and labeles
  Args:
     image_list_file: a .txt file with one /path/to/image per line
     label: optionally, if set label will be pasted after each line
  Returns:
     List with all filenames in file image_list_file
  '''  
  photo_records_to_txt(image_dir,txt_dir)
  #image_list_file = '/Users/phuongpham/Documents/infiles.txt'
  image_list_file = txt_dir
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
  resized = tf.image.resize_images(distorted_image, (45, 45))
  resized.set_shape([45,45,3])
  
  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
  #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)  
  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_whitening(distorted_image)
  return resized

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                  batch_size, shuffle):
  '''
  Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  '''
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  if shuffle:
    images, label_batch = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch_size,
                                                  capacity=min_queue_examples + 3 * batch_size,
                                                  min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          capacity=min_queue_examples + 3 * batch_size)
  tf.image_summary('images', images)
  #print (label_batch)
  return images, label_batch

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
'''
def inputs(data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = []
  labels = []
  #if eval_data==0:
  filenames, labels = extract_labels(data_dir_train,txt_dir_train,one_hot=True, num_classes=3)
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  
  else:
    filenames, labels = extract_labels(data_dir_eval,txt_dir_eval,one_hot=True, num_classes=3)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    txt_dir = txt_dir_eval
    
  #filenames, labels = extract_labels(one_hot=True, num_classes=3)
  #filenames = tf.train.match_filenames_once("/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/*.jpg")
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good3.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good4.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good5.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium3.jpeg']
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg']
  filename_queue = tf.train.string_input_producer(filenames,num_epochs=20)
  filename,read_input = read_image(filename_queue)
  reshaped_image = distorted_image(read_input)
  filenames, labels = extract_labels(data_dir_train,txt_dir_train,one_hot=True, num_classes=3)
  #filenames, labels = extract_labels(data_dir, one_hot=True, num_classes=3)
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
  #filename,read_input = read_image(filename_queue)
  #filenames,labels = _generate_image_and_label_batch(reshaped_image, labels,
                      #                                  min_queue_examples, batch_size,
                       #                                 shuffle=False)
  #return filename,reshaped_image
  return filenames,labels
'''

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string.
  Args:
    filename_and_label_tensor: A scalar string tensor.
  Returns:
    Two tensors: the decoded image, and the string label.
  """
  label = input_queue[1]
  file_contents = tf.read_file(input_queue[0])
  example = tf.image.decode_png(file_contents, channels=3)
  return example, label
  
#def inputs(eval,batch_size):
def inputs(eval,batch_size):
  '''
  Return a tensor of distorded image 
  '''

  filenames = []
  labels = []
  if not eval:
    print ('========TRAIN=========')
    filenames, labels = extract_labels(data_dir_train,txt_dir_train,one_hot=False, num_classes=3)
    #train_filenames, test_labels = extract_labels(data_dir_train,txt_dir_train,one_hot=False, num_classes=3)
    #print (labels)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    print ('========TEST=========')
    filenames, labels = extract_labels(data_dir_eval,txt_dir_eval,one_hot=True, num_classes=3)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  #filenames = tf.train.match_filenames_once("/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/*.jpg")
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good3.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good4.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good5.jpg', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/poor2.JPG', '/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/medium3.jpeg']
  #filenames = ['/Users/phuongpham/Documents/CanePhotoToKmut/7-9months/good2.jpg']
  filename_queue = tf.train.string_input_producer(filenames)
  filename,read_input = read_image(filename_queue)
  reshaped_image = distorted_image(read_input)
  my_labels = tf.constant(labels)
  images = tf.convert_to_tensor(filenames, dtype=tf.string)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  input_queue = tf.train.slice_input_producer([images, labels],shuffle=True)
  image, label = read_images_from_disk(input_queue)
  #print (label)
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
  batch_size = 20
  images,labels = _generate_image_and_label_batch(reshaped_image, label,
                                    min_queue_examples, batch_size,
                                    shuffle=False)
  #filenames, labels = extract_labels(data_dir_train,txt_dir_train,one_hot=True, num_classes=3)
  #filename,read_input = read_image(filename_queue)
  print (images)
  return images, labels

  
''' 
  for i in xrange(5):
    filename,img = sess.run(image)
    img = Image.fromarray(img, "RGB")
    img.save(os.path.join(data_dir_train,"foo"+str(i)+".jpg"))
'''
#MODEL
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1./math.sqrt(80)))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    #print ('CONV')
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='VALID')
    layer += biases
    #print ('layer')
    #print (layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    #print ('layer')
    #print (layer)
    layer = tf.nn.relu(layer)
    return layer, weights
def flatten_layer(layer):
  layer_shape = layer.get_shape()
  num_features = layer_shape[1:4].num_elements()
  layer_flat = tf.reshape(layer, [-1, num_features])
  return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
         num_inputs,     # Num. inputs from prev. layer.
         num_outputs,    # Num. outputs.
         use_relu=True): # Use Rectified Linear Unit (ReLU)?
  weights = new_weights(shape=[num_inputs, num_outputs])
  biases = new_biases(length=num_outputs)
  layer = tf.matmul(input, weights) + biases
  if use_relu:
    layer = tf.nn.relu(layer)
  return layer
'''
x_batch, y_true_batch = inputs(None, train_batch_size)
sess = tf.Session()
#sess.run(init)
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners()
sess.run(inputs(None, train_batch_size))
sess.close()
'''


def optimize():
  #image = extract_labels(one_hot = True, num_classes = 3)
  global total_iterations
  for i in range(total_iterations,total_iterations + 1):
    image = inputs(None,train_batch_size)
    tf.train.start_queue_runners(sess=sess)
    x_batch,y_true_batch = sess.run(image)
    #x_batch,y_true_batch = train_images, train_labels
    #print (train_labels)
    y_true_batch = np.array(y_true_batch)
    y_true_batch = dense_to_one_hot(y_true_batch, num_classes)
    #print (x_batch)
    #print(y_true_batch)
    #print(tf.argmax(y_true_batch, dimension=1))
    feed_dict_train = {x: x_batch, y_true: y_true_batch}
    #print (sess.run(y_true_cls, feed_dict=feed_dict_train))
    #print (sess.run(y_pred, feed_dict=feed_dict_train))
    #print (sess.run(y_pred_cls, feed_dict=feed_dict_train))
    sess.run(optimizer, feed_dict=feed_dict_train)
    sess.run(correct_prediction, feed_dict=feed_dict_train)
    sess.run(weights_conv1, feed_dict=feed_dict_train)
    my_accuracy = sess.run(accuracy, feed_dict=feed_dict_train)
    print ('training accuracy:', my_accuracy)
    #print(layer_conv1)
    #print(layer_flat)
    #print(layer_fc2)
  total_iterations += 1
  #msg = "Training Accuracy: {%}"
  #print(msg.format(acc))
  
def print_test_accuracy():
    
    # The ending index for the next batch is denoted j.
    
    # Get the images from the test-set between index i and j.
    image = inputs(True,train_batch_size)
    x_batch,y_true_batch = sess.run(image)
    #x_batch,y_true_batch = train_images, train_labels
    #print (train_labels)
    y_true_batch = np.array(y_true_batch)
    y_true_batch = dense_to_one_hot(y_true_batch, num_classes)
      
    # Create a feed-dict with these images and labels.
    feed_dict = {x: x_batch,y_true: y_true_batch}
    
    print ('Almost done')
    my_accuracy = sess.run(accuracy, feed_dict=feed_dict_train)
    print ('training accuracy:', my_accuracy)

with tf.Graph().as_default():
  x = tf.placeholder(tf.float32,shape=[None, 45,45,3], name='x')
  x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
  y_true = tf.placeholder(tf.float32,shape=[None, 3], name='y_true')
  y_true_cls = tf.argmax(y_true, dimension=1)
  #print(y_true_cls.get_shape())
  layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                        num_input_channels=num_channels,
                        filter_size=6,
                        num_filters=32,
                        use_pooling=True)
  #print ('layer 1')
  layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                        num_input_channels=num_filters1,
                        filter_size=filter_size2,
                        num_filters=num_filters2,
                        use_pooling=True)
  #print ('layer 2')
  layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                        num_input_channels=num_filters2,
                        filter_size=filter_size3,
                        num_filters=num_filters3,
                        use_pooling=False)  
  #print ('layer 3')
  layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                        num_input_channels=num_filters3,
                        filter_size=filter_size4,
                        num_filters=num_filters4,
                        use_pooling=True)  
  #print ('layer 4')
  layer_flat, num_features = flatten_layer(layer_conv4)
  layer_fc1 = new_fc_layer(input=layer_flat,
               num_inputs=num_features,
               num_outputs=fc_size,
               use_relu=True)
  layer_fc2 = new_fc_layer(input=layer_fc1,
               num_inputs=fc_size,
               num_outputs=num_classes,
               use_relu=False)
  y_pred = tf.nn.softmax(layer_fc2)
  y_pred_cls = tf.argmax(y_pred, dimension=1)
  #print(y_pred_cls)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                labels=y_true)
  cost = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
  #print ('CORRECT PREDICTION')
  #print (correct_prediction)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #print (accuracy)
  session = tf.InteractiveSession() 

  train_batch_size = 1
  test_batch_size = 20
  #train_images, train_labels = inputs(None,train_batch_size)
  #test_images, test_labels = inputs(True,train_batch_size)
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  total_iterations = 0
  optimize()
  print_test_accuracy()

    
"""
test_batch_size = 3
def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
  num_test = file_len(txt_dir_eval)
  # Allocate an array for the predicted classes which
  # will be calculated in batches and filled into this array.
  cls_pred = np.zeros(shape=num_test, dtype=np.int)

  # Now calculate the predicted classes for the batches.
  # We will just iterate through all the batches.
  # There might be a more clever and Pythonic way of doing this.

  # The starting index for the next batch is denoted i.
  i = 0
  while i < num_test:
    # The ending index for the next batch is denoted j.
    j = min(i + test_batch_size, num_test)
    images = test_images[i:j, :]
    labels = test_labels[i:j, :]
    # Create a feed-dict with these images and labels.
    feed_dict = {x: images,
           y_true: labels}

    # Calculate the predicted class using TensorFlow.
    cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

    # Set the start-index for the next batch to the
    # end-index of the current batch.
    i = j
# Convenience variable for the true class-numbers of the test-set.
cls_true = data.test.cls

# Create a boolean array whether each image is correctly classified.
correct = (cls_true == cls_pred)

# Calculate the number of correctly classified images.
# When summing a boolean array, False means 0 and True means 1.
correct_sum = correct.sum()

# Classification accuracy is the number of correctly classified
# images divided by the total number of images in the test-set.
acc = float(correct_sum) / num_test

# Print the accuracy.
msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
print(msg.format(acc, correct_sum, num_test))
"""
'''
with tf.Graph().as_default():
  #image = extract_labels(one_hot = True, num_classes = 3)
  image = inputs(None, train_batch_size)
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  sess.run(image)
'''
#y_pred.eval(feed_dict = {x: x_batch.eval(session=sess), y_true: y_true_batch.eval(session=sess)},session=sess)
#print (y_true_batch.eval(session=sess))
'''
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
x = tf.placeholder(tf.float32,shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.InteractiveSession() 
sess = tf.Session()
sess.run(tf.initialize_all_variables()) 
train_batch_size = 30
# Counter for total number of iterations performed so far.
total_iterations = 0
num_iterations = 1 
#def optimize(num_iterations):
#with sess.as_default():
    # Ensure we update the global variable rather than a local copy.
    #global total_iterations
    #for i in range(total_iterations,
    #              total_iterations + num_iterations):
        
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        # x_batch, y_true_batch = images.train.next_batch(train_batch_size)
x_batch, y_true_batch = inputs(None, train_batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
feed_dict_train = {x: x_batch.eval(), y_true: y_true_batch.eval()}
        
        
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
session.run(optimizer, feed_dict=feed_dict_train)
        
        # Print status every 100 iterations.
        #if i % 100 == 0:
            # Calculate the accuracy on the training-set.
acc = session.run(accuracy, feed_dict=feed_dict_train)
            #feed_dict_train = accuracy.eval(session=session, feed_dict={x:x_batch.eval(session=sess), y_true: y_true_batch(session=sess)})
            # Message for printing.
msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            
            # Print it.
print(msg.format(acc))
    # Update the total number of iterations performed.
    #total_iterations += num_iterations
'''
'''
with tf.Graph().as_default():
  session.run(optimize(num_iterations=1)) 
 
with tf.Graph().as_default():
  #image = extract_labels(one_hot = True, num_classes = 3)
  op = optimize(num_iterations=1)
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  session.run(op)

optimize(num_iterations=1)

#calculate cross entropy
def loss(logits, labels):

    #logits: result from the last fc layer.
    #label: Labels from distorted_inputs or inputs().  1-D tensor of [batch_size] size.
    
    # Calculate the average cross entropy loss across the batch.
    # Measures the probability error in discrete classification tasks in which the classes are mutually exclusive
    cross_entropy = tf.nn.spare_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y_true)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('loss', cross_entropy_mean)
    return tf.add_n(tf.get_collection('loss'), name='total_loss')

def training(learning rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    losses = tf.get_collection('loss')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def training(total_loss, global_step):
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

#Create the model
x = tf.placeholder(tf.float32, shape=[None, img_size_flat]]
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
y_true_cls = tf.argmax(y_true, dimension=1)

total_iterations = 0
def train(num_iterations):
    global total_iterations
    for i in range(total_iterations,total_iterations+num_iterations):
        batch = mnist.train.next_batch(num_iterations)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
                       
    
with tf.Graph().as_default():
  #image = extract_labels(one_hot = True, num_classes = 3)
  image = inputs(None)
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  for i in xrange(5):
    filename,img = sess.run(image)
    img = Image.fromarray(img, "RGB")
    if filename.find('good') != -1:
      img.save(os.path.join(data_dir_train,"good"+str(i)+".jpg"))
    if filename.find('medium') != -1:
      img.save(os.path.join(data_dir_train,"medium"+str(i)+".jpg"))
    if filename.find('poor') != -1:
      img.save(os.path.join(data_dir_train,"poor"+str(i)+".jpg"))

'''

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