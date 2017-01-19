#!/usr/bin/python


print ('layer 1')
print (layer_conv1)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
											num_input_channels=num_filters1,
											filter_size=filter_size2,
											num_filters=num_filters2,
											use_pooling=True)
print ('layer 2')
print (layer_conv2)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
											num_input_channels=num_filters2,
											filter_size=filter_size3,
											num_filters=num_filters3,
											use_pooling=False)  
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
											num_input_channels=num_filters3,
											filter_size=filter_size4,
											num_filters=num_filters4,
											use_pooling=True)  