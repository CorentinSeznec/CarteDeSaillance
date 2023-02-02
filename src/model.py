import tensorflow as tf

# using Mobilenet neural network 

# process stuff

base_resnet_net = tf.keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.


# Freeze the base model

base_resnet_net.trainable = False


# Create a new model on top

inputs = tf.keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_resnet_net(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = tf.keras.layers.Dense(1)(x)
model= tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
                loss="categorical_crossentropy")

model.summary()



# train et entrainer sur le 1/5 de test2, on veut voir ce que ça améliore entrainer sur test2 et test sur test2 et test1 
# on entraine sucessivement les 5 batchs pour voir l'evolution des res sur test1 et le batch qu'on vient d'utiliser
# resNet
# pdf rapport : temps,conso electique et carbone, discussion res, valeur accuracy tableau, confusion matrix, courbes train et validation etc + archive avec le code 
# carbon tracker

# prochaine etape: charger test pour le mettre dans la validation et mettre des checkspoints