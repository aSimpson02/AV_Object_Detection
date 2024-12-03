#importing relevant lubarries
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from load_data import data_preprocessing, data_augmentation
from CNN_model import building_model, training_model, save_model
from keras.utils import to_categorical


#checkng paths
path = 'KittiDataset'
train_image_path = os.path.join(path, 'image_2', 'training')
train_label_path = os.path.join(path, 'label_2')
test_image_path = os.path.join(path, 'image_2', 'testing')


paths = [train_image_path, train_label_path, test_image_path]
for p in paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"path does not exist: {p}")
    print(f"Verified path: {p}")


#loading and preprocessing teh data
print("loading and preprocessing training data...")
train_images, train_labels = data_preprocessing(train_image_path, train_label_path)

print("loading and preprocessing testing data...")
test_images, _ = data_preprocessing(test_image_path)

#encoding labels
class_mapping = {
    'car': 0, 'irrelevant': 1, 'pedestrian': 2, 'bike': 3,
    'lorry': 4, 'bus': 5, 'traffic Light': 6,
    'road Sign': 7, 'motorbike': 8
}

train_labels = [
    class_mapping.get(label.split()[0], -1) if label is not None else -1
    for label in train_labels
]


#checking for any invalid labels
if -1 in train_labels:
    print("Warning: Some labels not mapped to class ID.")

train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels, num_classes=len(class_mapping))

# saving preprocessed data
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)

#augmentation
print("data augmentation...")
train_data = data_augmentation(train_images, train_labels)

#splitting up data
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

#using function in cnn_model.py file to build and train the model
model = building_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("training model...")
trained_model = training_model(model, X_train, y_train, X_val, y_val, epochs=10)

#saving the model
save_model(model, filename="my_model.keras")
print("training complete. Model saved.")
