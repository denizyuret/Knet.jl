import tensorflow as tf
validation_dir="fooval/val" #"/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val" #"val"
BATCH_SIZE = 32
IMG_SIZE = (224,224)

#model = tf.keras.applications.mobilenet
#pretrained = tf.keras.applications.MobileNet

model = tf.keras.applications.mobilenet_v2
pretrained = tf.keras.applications.MobileNetV2

#model = tf.keras.applications.mobilenet_v3
#pretrained = tf.keras.applications.MobileNetV3Small
#pretrained = tf.keras.applications.MobileNetV3Large

ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    i = model.preprocess_input(i)
    return (i, label)

# Preprocess the images
ds = ds.map(resize_with_crop)

# Compile the model
pretrained_model = pretrained(include_top=True, weights='imagenet')
pretrained_model.trainable = False
pretrained_model.compile(optimizer='adam', 
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                         metrics=['accuracy'])
decode_predictions = model.decode_predictions

# Print Accuracy
result = pretrained_model.evaluate(ds)
print(dict(zip(pretrained_model.metrics_names, result)))
