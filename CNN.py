#Part-1: Building the CNN
#import packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Adding the different layers
#step-1-Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation ='relu'))

#step-2-Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

#step-3-Flattening-1D-Array
classifier.add(Flatten())

#step-4-Full Connection
classifier.add(Dense(output_dim = 128 , activation ='relu'))
classifier.add(Dense(output_dim = 1 , activation ='sigmoid'))

#compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics =['accuracy'])

#Part-2-fitting the CNN 
#image preprocessing using keras and image augmentation 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=2000)

#part-3-predicting new
import numpy as np
from keras.preprocessing.image import img_to_array, load_img;
test_img = load_img('cat_or_dog_2.jpg',target_size=(64,64))
test_img = img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result=classifier.predict(test_img)
training_set.class_indices
if result[0][0] == 1:
      prediction = 'dog'
else:
      prediction = 'cat'      






