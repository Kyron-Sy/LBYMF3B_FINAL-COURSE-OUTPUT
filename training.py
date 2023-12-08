from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 200, 200

train_data_dir = 'data/training'
validation_data_dir = 'data/validation'
nb_train_samples = 276
nb_validation_samples = 12
epochs = 32
batch_size = 5
num_classes = 6

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print("img data format channels_first")
else:
    input_shape = (img_width, img_height, 3)

# ...
model = Sequential()
model.add(Conv2D(32, (6, 6), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (6, 6)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (6, 6)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))  # Replace num_classes with the number of classes
model.add(Activation('softmax'))  # Use softmax for multiclass classification

model.compile(loss='categorical_crossentropy',  # Use categorical_crossentropy for multiclass
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    class_mode='categorical')
    class_mode='categorical')

print("train_generator")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    class_mode='categorical')
    class_mode='categorical')

print("validation_generator")

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

print("fit_generator")
#model.save('model.h5')
#model.save_weights('weights.h5')
model.save('model.keras')
model.save_weights('weights.keras')

print("saved model and weights")
