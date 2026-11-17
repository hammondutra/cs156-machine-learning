"""CS156 Session 20 — MobileNetV2 transfer learning and PCA feature visualization.
Run: python src/session20_transfer_learning.py
Optional: CS156_EPOCHS=1 CS156_FINE_EPOCHS=1 CS156_NO_SHOW=1 python src/session20_transfer_learning.py
GPU recommended; first run downloads the public cats/dogs dataset and ImageNet weights.
"""
import os
if os.getenv("CS156_NO_SHOW") == "1":
    import matplotlib
    matplotlib.use("Agg")

# %% [markdown]
# ##### Copyright 2019 The TensorFlow Authors.

# %% [markdown]
# **CS156 Session 20 — Transfer Learning and Fine-tuning**
# 
# **Main learning notebook** — TensorFlow/Keras MobileNetV2 transfer-learning tutorial, with the CS156 cats/dogs PCA extension. The companion notes notebook contains the shorter PCW answers. The original student's executed Colab export is available separately; outputs in this GitHub version are cleared for size and reproducibility. Run all cells on a GPU-enabled Colab runtime.

# %% [markdown]
# # Transfer learning and fine-tuning

# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/images/transfer_learning"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb?force_kitty_mode=1&force_corgi_mode=1"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/images/transfer_learning.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# %% [markdown]
# In this tutorial, you will learn how to classify images of cats and dogs by using transfer learning from a pre-trained network.
# 
# A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task.
# 
# The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.
# 
# In this notebook, you will try two ways to customize a pretrained model:
# 
# 1. Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset.
# 
#  You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.
# 
# 1. Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.
# 
# You will follow the general machine learning workflow.
# 
# 1. Examine and understand the data
# 1. Build an input pipeline, in this case using Keras ImageDataGenerator
# 1. Compose the model
#    * Load in the pretrained base model (and pretrained weights)
#    * Stack the classification layers on top
# 1. Train the model
# 1. Evaluate model
# 

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# ### Data download

# %% [markdown]
# In this tutorial, you will use a dataset containing several thousand images of cats and dogs. Download and extract a zip file containing the images, then create a `tf.data.Dataset` for training and validation using the `tf.keras.utils.image_dataset_from_directory` utility. You can learn more about loading images in this [tutorial](https://www.tensorflow.org/tutorials/load_data/images).

# %%
import os
import zipfile
import tensorflow as tf
_URL = "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=_URL, extract=False)
DATA_DIR = os.environ.get("CS156_DATA_DIR", "/content" if os.path.isdir("/content") else os.path.expanduser("~/.cache/cs156"))
os.makedirs(DATA_DIR, exist_ok=True)
with zipfile.ZipFile(path_to_zip, "r") as archive:
    archive.extractall(DATA_DIR)
PATH = os.path.join(DATA_DIR, "cats_and_dogs_filtered")
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")
assert os.path.isdir(train_dir) and os.path.isdir(validation_dir), PATH
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# %%
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

# %% [markdown]
# Show the first nine images and labels from the training set:

# %%
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %% [markdown]
# As the original dataset doesn't contain a test set, you will create one. To do so, determine how many batches of data are available in the validation set using `tf.data.experimental.cardinality`, then move 20% of them to a test set.

# %%
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# %%
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

# %% [markdown]
# ### Configure the dataset for performance

# %% [markdown]
# Use buffered prefetching to load images from disk without having I/O become blocking. To learn more about this method see the [data performance](https://www.tensorflow.org/guide/data_performance) guide.

# %%
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# ### Use data augmentation

# %% [markdown]
# When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity by applying random, yet realistic, transformations to the training images, such as rotation and horizontal flipping. This helps expose the model to different aspects of the training data and reduce [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). You can learn more about data augmentation in this [tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).

# %%
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# %% [markdown]
# Note: These layers are active only during training, when you call `Model.fit`. They are inactive when the model is used in inference mode in `Model.evaluate`, `Model.predict`, or `Model.call`.

# %% [markdown]
# Let's repeatedly apply these layers to the same image and see the result.

# %%
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

# %% [markdown]
# ### Rescale pixel values
# 
# In a moment, you will download `tf.keras.applications.MobileNetV2` for use as your base model. This model expects pixel values in `[-1, 1]`, but at this point, the pixel values in your images are in `[0, 255]`. To rescale them, use the preprocessing method included with the model.

# %%
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# %% [markdown]
# Note: Alternatively, you could rescale pixel values from `[0, 255]` to `[-1, 1]` using `tf.keras.layers.Rescaling`.

# %%
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# %% [markdown]
# Note: If using other `tf.keras.applications`, be sure to check the API doc to determine if they expect pixels in `[-1, 1]` or `[0, 1]`, or use the included `preprocess_input` function.

# %% [markdown]
# ## Create the base model from the pre-trained convnets
# You will create the base model from the **MobileNet V2** model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a wide variety of categories like `jackfruit` and `syringe`. This base of knowledge will help us classify cats and dogs from our specific dataset.
# 
# First, you need to pick which layer of MobileNet V2 you will use for feature extraction. The very last classification layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful. Instead, you will follow the common practice to depend on the very last layer before the flatten operation. This layer is called the "bottleneck layer". The bottleneck layer features retain more generality as compared to the final/top layer.
# 
# First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the **include_top=False** argument, you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.

# %%
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# %% [markdown]
# This feature extractor converts each `160x160x3` image into a `5x5x1280` block of features. Let's see what it does to an example batch of images:

# %%
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# %% [markdown]
# ## Feature extraction
# In this step, you will freeze the convolutional base created from the previous step and to use as a feature extractor. Additionally, you add a classifier on top of it and train the top-level classifier.

# %% [markdown]
# ### Freeze the convolutional base

# %% [markdown]
# It is important to freeze the convolutional base before you compile and train the model. Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. MobileNet V2 has many layers, so setting the entire model's `trainable` flag to False will freeze all of them.

# %%
base_model.trainable = False

# %% [markdown]
# ### Important note about BatchNormalization layers
# 
# Many models contain `tf.keras.layers.BatchNormalization` layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial.
# 
# When you set `layer.trainable = False`, the `BatchNormalization` layer will run in inference mode, and will not update its mean and variance statistics.
# 
# When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, you should keep the BatchNormalization layers in inference mode by passing `training = False` when calling the base model. Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.
# 
# For more details, see the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).

# %%
# Let's take a look at the base model architecture
base_model.summary()

# %% [markdown]
# ### Add a classification head

# %% [markdown]
# To generate predictions from the block of features, average over the spatial `5x5` spatial locations, using a `tf.keras.layers.GlobalAveragePooling2D` layer to convert the features to  a single 1280-element vector per image.

# %%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# %% [markdown]
# Apply a `tf.keras.layers.Dense` layer to convert these features into a single prediction per image. You don't need an activation function here because this prediction will be treated as a `logit`, or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.

# %%
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# %% [markdown]
# Build a model by chaining together the data augmentation, rescaling, `base_model` and feature extractor layers using the [Keras Functional API](https://www.tensorflow.org/guide/keras/functional). As previously mentioned, use `training=False` as our model contains a `BatchNormalization` layer.

# %%
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# %%
model.summary()

# %% [markdown]
# The 8+ million parameters in MobileNet are frozen, but there are 1.2 thousand _trainable_ parameters in the Dense layer. These are divided between two `tf.Variable` objects, the weights and biases.

# %%
print("Trainable tensors:", len(model.trainable_variables))

# %%
try:
    tf.keras.utils.plot_model(model, show_shapes=True)
except (ImportError, OSError) as exc:
    print("Skipping Graphviz architecture plot:", exc)

# %% [markdown]
# ### Compile the model
# 
# Compile the model before training it. Since there are two classes and a sigmoid oputput, use the `BinaryAccuracy`.

# %%
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

# %% [markdown]
# ### Train the model
# 
# After training for 10 epochs, you should see ~96% accuracy on the validation set.
# 

# %%
initial_epochs = int(os.environ.get("CS156_EPOCHS", "10"))

loss0, accuracy0 = model.evaluate(validation_dataset)

# %%
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# %%
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# %% [markdown]
# ### Learning curves
# 
# Let's take a look at the learning curves of the training and validation accuracy/loss when using the MobileNetV2 base model as a fixed feature extractor.

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %% [markdown]
# Note: If you are wondering why the validation metrics are clearly better than the training metrics, the main factor is because layers like `tf.keras.layers.BatchNormalization` and `tf.keras.layers.Dropout` affect accuracy during training. They are turned off when calculating validation loss.
# 
# To a lesser extent, it is also because training metrics report the average for an epoch, while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.

# %% [markdown]
# ## Fine tuning
# In the feature extraction experiment, you were only training a few layers on top of an MobileNetV2 base model. The weights of the pre-trained network were **not** updated during training.
# 
# One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.
# 
# Note: This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.
# 
# Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features that generalize to almost all types of images. As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

# %% [markdown]
# ### Un-freeze the top layers of the model
# 

# %% [markdown]
# All you need to do is unfreeze the `base_model` and set the bottom layers to be un-trainable. Then, you should recompile the model (necessary for these changes to take effect), and resume training.

# %%
base_model.trainable = True

# %%
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# %% [markdown]
# ### Compile the model
# 
# As you are training a much larger model and want to readapt the pretrained weights, it is important to use a lower learning rate at this stage. Otherwise, your model could overfit very quickly.

# %%
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

# %%
model.summary()

# %%
print("Trainable tensors:", len(model.trainable_variables))

# %% [markdown]
# ### Continue training the model

# %% [markdown]
# If you trained to convergence earlier, this step will improve your accuracy by a few percentage points.

# %%
fine_tune_epochs = int(os.environ.get("CS156_FINE_EPOCHS", "10"))
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=len(history.epoch),
                         validation_data=validation_dataset)

# %% [markdown]
# Let's take a look at the learning curves of the training and validation accuracy/loss when fine-tuning the last few layers of the MobileNetV2 base model and training the classifier on top of it. The validation loss is much higher than the training loss, so you may get some overfitting.
# 
# You may also get some overfitting as the new training set is relatively small and similar to the original MobileNetV2 datasets.
# 

# %% [markdown]
# After fine tuning the model nearly reaches 98% accuracy on the validation set.

# %%
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# %%
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %% [markdown]
# ### Evaluation and prediction

# %% [markdown]
# Finally you can verify the performance of the model on new data using test set.

# %%
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# %% [markdown]
# And now you are all set to use this model to predict if your pet is a cat or dog.

# %%
# Retrieve a batch of images from the test set
image_batch, label_batch = next(test_dataset.as_numpy_iterator())
predictions = model.predict_on_batch(image_batch).flatten()
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")

# %% [markdown]
# ## Summary
# 
# * **Using a pre-trained model for feature extraction**:  When working with a small dataset, it is a common practice to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier get updated during training.
# In this case, the convolutional base extracted all the features associated with each image and you just trained a classifier that determines the image class given that set of extracted features.
# 
# * **Fine-tuning a pre-trained model**: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning.
# In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.
# 
# To learn more, visit the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).
# 

# %% [markdown]
# ## CS156 — Extension Question (Question 3): Visualizing Embeddings with PCA
# 
# **Attempting the extension.** The task is to calculate 1,280-dimensional pooled final-layer vectors for all original images from the **unfine-tuned ImageNet** MobileNetV2 backbone and project them with PCA into 2D. No cat/dog classification head is trained on the pretrained extractor.

# %%
# Extension 3A — Use original, unfine-tuned pretrained MobileNetV2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
embedding_base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
embedding_base.trainable = False
embedding_inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(embedding_inputs)
x = embedding_base(x, training=False)
embedding_outputs = tf.keras.layers.GlobalAveragePooling2D(name="feature_batch_average")(x)
feature_extractor = tf.keras.Model(embedding_inputs, embedding_outputs)
print("Embedding dimensions:", feature_extractor.output_shape[-1])
print("Trainable CNN weights:", len(feature_extractor.trainable_weights))

# %% [markdown]
# ### Extract a feature vector for every image
# 
# Load the 2,000 training images and 1,000 validation images with `shuffle=False`. The full tutorial has already fine-tuned a separate base model above, so this feature extractor is deliberately reinstantiated with original ImageNet weights.

# %%
# Extension 3B — Extract one 1,280-D vector for every image (no model fitting)
train_for_embeddings = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_for_embeddings = tf.keras.utils.image_dataset_from_directory(
    validation_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
assert train_for_embeddings.class_names == validation_for_embeddings.class_names
embedding_class_names = train_for_embeddings.class_names
all_images = train_for_embeddings.concatenate(validation_for_embeddings)
embeddings = feature_extractor.predict(all_images, verbose=1)
true_labels = np.concatenate([labels.numpy() for _, labels in all_images])
assert len(embeddings) == len(true_labels)
print("Embeddings shape:", embeddings.shape)
print("Classes:", embedding_class_names)

# %% [markdown]
# ### Visualize 1,280-D embeddings with PCA
# 
# PCA is label-agnostic: it maximizes variance, not class separation.

# %%
# Extension 3C — PCA from 1,280 to 2 dimensions
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(10, 7))
for i, name in enumerate(embedding_class_names):
    points = embeddings_2d[true_labels == i]
    ax.scatter(points[:, 0], points[:, 1], s=16, alpha=.55, label=name)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} explained variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} explained variance)")
ax.set_title("Unfine-tuned MobileNetV2: cats vs dogs in PCA space")
ax.legend(title="Actual label")
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()
print(f"First two PCs explained variance: {sum(pca.explained_variance_ratio_):.1%}")

# %% [markdown]
# ### Optional 2D linear probe
# 
# Use stratified held-out evaluation to fit and evaluate a line in **PCA space**, without any gradient updates to the CNN.

# %%
# Extension 3D — Fit a 2D linear probe; DO NOT fine-tune the CNN
X_train, X_holdout, y_train, y_holdout = train_test_split(
    embeddings_2d, true_labels, test_size=.2, stratify=true_labels, random_state=42)
linear_probe = LogisticRegression(max_iter=1000)
linear_probe.fit(X_train, y_train)
print(f"2D linear probe held-out accuracy: {linear_probe.score(X_holdout, y_holdout):.1%}")
fig, ax = plt.subplots(figsize=(10, 7))
for i, name in enumerate(embedding_class_names):
    points = embeddings_2d[true_labels == i]
    ax.scatter(points[:, 0], points[:, 1], s=16, alpha=.45, label=name)
w0, w1 = linear_probe.coef_[0]
bias = linear_probe.intercept_[0]
if abs(w1) > 1e-12:
    line_x = np.linspace(embeddings_2d[:,0].min(), embeddings_2d[:,0].max(), 200)
    ax.plot(line_x, -(w0*line_x+bias)/w1, "k--", linewidth=2, label="Linear probe boundary")
elif abs(w0) > 1e-12:
    ax.axvline(-bias/w0, color="k", linestyle="--", linewidth=2, label="Linear probe boundary")
ax.set_xlim(embeddings_2d[:,0].min(), embeddings_2d[:,0].max())
ax.set_ylim(embeddings_2d[:,1].min(), embeddings_2d[:,1].max())
ax.set_xlabel("First principal component")
ax.set_ylabel("Second principal component")
ax.set_title("Linear separability of pretrained embeddings (2D PCA view)")
ax.legend()
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation
# 
# The completed Colab run showed cats and dogs occupying largely distinct regions in the first two principal components, with an approximately straight boundary dividing most points. A linear classifier can therefore reuse general visual features from pretrained MobileNetV2 rather than learning those features from scratch. The two-dimensional projection is only a visualization: overlap here does not preclude separation in the original 1,280-dimensional embedding space, and a held-out **2D** probe score is not the same as a test-set evaluation of the fine-tuned CNN.
