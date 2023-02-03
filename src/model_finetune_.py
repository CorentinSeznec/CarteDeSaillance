import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import *
import matplotlib.pyplot as plt
from carbontracker.tracker import CarbonTracker


tf.config.list_physical_devices('GPU')


## Parameters

isSalliancyMap = sys.argv[1] # 1 for salliancy
if isSalliancyMap:
        folder = '/Saillances'

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



nbr_subdataset = 5 # number of subdataset of test1
elmt_per_split = nbr_test_img1 // nbr_subdataset
batch_size = 50
epochs = 10
model_name = "model_finetune"
model_to_load = "regular_model"



## Start
print("\n\nSTART\n")

## import model
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

# Load model with weights compute with the regular_model script
model.load_weights('..'+folder+'/checkpoints/checkpoint_'+ model_to_load+".ckpt")

print("\nMODEL LOADED")


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


checkpoint_finetuned_filepath = '..'+folder+'/checkpoints_finetuned/checkpoint_'+ model_name+".ckpt"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_finetuned_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


## Creation of datagenerator
print("\nDATAGENERATOR 1 & 2 CREATION")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen1 = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen2 = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

test_generator1 = test_datagen1.flow_from_directory(
        test_dir1,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

test_generator2 = test_datagen2.flow_from_directory(
        test_dir2,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


# Unfreeze model
print("\nUNFREEZE THE END OF THE MODEL...")
for layer in model.layers[:-1]:

        if layer.name == "vgg16":
                
                nb_sublayer = len(layer.layers)
                print("nb_sublayer", nb_sublayer)
                for sublayer in layer.layers[:-2]:
                        print("modified to non learnable", sublayer.name)
                        sublayer.trainable = False
                for sublayer in layer.layers[-2:]:
                        print("modified to learnable", sublayer.name)
                        sublayer.trainable = True
        else:
                print("modified to non learnable", layer.name)
                layer.trainable = False

for layer in model.layers[-1:]:
        print("modified to learnable", layer.name)
        layer.trainable = True


## Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="categorical_crossentropy",
                metrics=['accuracy'])


model.summary()



## results with model imported
# print("\nCOMPUTE RESULTS WITH LOADED MODEL")
# AccuracyOn_test_generator1 = model.evaluate(test_generator1, batch_size=batch_size)
# AccuracyOn_test_generator2 = model.evaluate(test_generator2, batch_size=batch_size)



print("\nSPLITTING DATAGENERATOR1 INTO SUBDATASETS")
# Convert datagenerator into dataset
ds_counter = tf.data.Dataset.from_generator(
                lambda: test_datagen1.flow_from_directory(
                test_dir1,
                target_size=(150, 150),
                batch_size=elmt_per_split,
                class_mode='categorical'), 
                output_types=(tf.float32, tf.float32), 
                output_shapes=([elmt_per_split,150,150,3], [elmt_per_split,9])
        )

ds_counter.element_spec

print("\nCOMPUTE RESULTS WITH FINETUNEED MODEL")
history_saved = [0]
val_history_saved = [0]
for (iteration, (images, label)) in enumerate(ds_counter.take(nbr_subdataset)):
        
        print("\nIteration: "+str(iteration))

        # Convert subdataset into datagenerator
        data_gen1 = tf.keras.preprocessing.image.ImageDataGenerator()
        test_generator1_subdivided = data_gen1.flow(images, label)

        history = model.fit(
                test_generator1_subdivided,
                steps_per_epoch= elmt_per_split//batch_size,  
                epochs=epochs,
                validation_data=test_generator2,
                callbacks = [model_checkpoint_callback, CarbonTrackerCallback()],
                batch_size=batch_size,
                validation_steps = nbr_test_img2//batch_size,
                verbose=1
        )

        # load the best?
        model.load_weights(checkpoint_finetuned_filepath)

        # get accuracy at each iteration

        AccuracyOn_test_generator1 = model.evaluate(test_generator1, batch_size=batch_size)
        AccuracyOn_test_generator2 = model.evaluate(test_generator2, batch_size=batch_size)
        
        for acc in history.history['val_accuracy']:
              val_history_saved.append(acc)  
        for acc in history.history['accuracy']:
              history_saved.append(acc)  

        print("history_saved", history_saved)
        print("val_history_saved", val_history_saved)


# save plot

plt.plot(range(epochs*nbr_subdataset+1), history_saved, label ='Acc on test1')
plt.plot(range(epochs*nbr_subdataset+1), val_history_saved, label = 'Val Acc on test2')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Acc and Val Acc on test1 and test2 during Finetuning")
plt.savefig(".."+folder+"/ValAccuracyFinetuning2.png")



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


