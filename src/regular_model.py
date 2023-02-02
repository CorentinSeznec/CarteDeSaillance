import os
import sys
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import *
from carbontracker.tracker import CarbonTracker
import matplotlib.pyplot as plt
tf.config.list_physical_devices('GPU')


from tensorflow import keras
# model = keras.models.load_model('path/to/location')

isSalliancyMap = sys.argv[1] # 1 for salliancy
if isSalliancyMap:
        folder = '/Salliances'

else:
        folder = '/BB'

train_dir = ".."+folder+"/train1/"
test_dir1 = ".."+folder+"/test1/"
test_dir2 = ".."+folder+"/test2/"

nbr_train_img = 0
for root_dir, cur_dir, files in os.walk(train_dir):
    nbr_train_img += len(files)

nbr_test_img1 = 0
for root_dir, cur_dir, files in os.walk(test_dir1):
    nbr_test_img1 += len(files)

nbr_test_img2 = 0
for root_dir, cur_dir, files in os.walk(test_dir2):
    nbr_test_img2 += len(files)




batch_size = 50
epochs = 2
model_name = "regular_model"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

test_generator1 = test_datagen.flow_from_directory(
        test_dir1,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

test_generator2 = test_datagen.flow_from_directory(
        test_dir2,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


## Create model
base_resnet_net = tf.keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

# Unfreeze the base model
base_resnet_net.trainable = True

# Create a new model on top
inputs = tf.keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_resnet_net(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = tf.keras.layers.Dense(9,  activation = 'softmax')(x)
model= tf.keras.Model(inputs, outputs)

#optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=['accuracy'])


model.summary()


checkpoint_filepath = '..'+folder+'/checkpoints/checkpoint_'+ model_name+".ckpt"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

## Carbone tracker callbacks
class CarbonTrackerCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
                print("Start tracking")
                # Initialize the Carbon Tracker module
                self.tracker = CarbonTracker(epochs=epochs)

        def on_epoch_begin(self, epoch, logs=None):             
                self.tracker.epoch_start()
        
        def on_epoch_end(self, epoch, logs=None):
                # Call the Carbon Tracker module after each epoch
                self.tracker.epoch_end()

        def on_train_end(self, logs=None):
                print("Stop tracking")
                self.tracker.stop()



history = model.fit(
      x= train_generator,
      steps_per_epoch= nbr_train_img // batch_size,  # 10000 images = batch_size * steps      nb // batch_size
      epochs=epochs,
      validation_data=test_generator2,
      callbacks=[model_checkpoint_callback, CarbonTrackerCallback()],
      batch_size=batch_size,
      validation_steps = nbr_test_img2 // batch_size,
      verbose=1)



results1 = model.evaluate(test_generator1, batch_size=batch_size)
results2 = model.evaluate(test_generator2, batch_size=batch_size)

history_list = [0]
val_history_list = [0]


for acc in history.history['val_accuracy']:
        val_history_list.append(acc)  
for acc in history.history['accuracy']:
        history_list.append(acc)  

print("\nhistory",history_list)
print("\nval_history",val_history_list)

plt.plot(range(epochs+1), history_list, label="Accuracy")

plt.plot(range(epochs+1), val_history_list, label="Val Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Acc and val Acc on test 1 and test2 during training")
plt.legend()
plt.savefig(".."+folder+"/ValAccuracyTraining.png")



#Confution Matrix and Classification Report
print("\nTEST1")
Y_pred_1 = model.predict_generator(test_generator1, nbr_test_img1 // batch_size+1)
y_pred = np.argmax(Y_pred_1, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator1.classes, y_pred))
print('Classification Report')
target_names = list(train_generator.class_indices.keys())
print(classification_report(test_generator1.classes, y_pred, target_names=target_names))

print("\nTEST2")
Y_pred_2 = model.predict_generator(test_generator2, nbr_test_img2 // batch_size+1)
y_pred = np.argmax(Y_pred_2, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator2.classes, y_pred))
print('Classification Report')
target_names = list(train_generator.class_indices.keys())
print(classification_report(test_generator2.classes, y_pred, target_names=target_names))
