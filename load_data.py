#Importing relevant libraries:::
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import os



#upload data here 
#then process it before augmentation + model training 
#paths to the dataset in downloads folder
path = 'KittiDataset'

#looking into folders/data in the path of the dataset
train_image_path = os.path.join(path, 'image_2', 'training')
train_label_path = os.path.join(path, 'label_2')
train_calib_path = os.path.join(path, 'calib', 'training')

test_image_path = os.path.join(path, 'image_2', 'testing')
test_calib_path = os.path.join(path, 'calib', 'testing')



#verifying data here
paths = [train_image_path, train_label_path, train_calib_path, test_image_path, test_calib_path]
for p in paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path does not exist: {p}")
    else:
        print(f"Verified path: {p}")


#Data Preprocessing:::
#add in data here, augmentation here too
def data_preprocessing(image_path, label_path=None):
    print(f"Processing images in: {image_path}")
    if not isinstance(image_path, str):
        raise TypeError(f"Expected a string for image_path, but got {type(image_path)}")

    images, labels = [], []

    #lopping through to find all images in folders 
    for file_name in os.listdir(image_path):
        #print(f"Contents of {image}: {os.listdir(image)}")
        if file_name.lower().endswith('.png'):
            #loading
            img_file = os.path.join(image_path, file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(f"Warning: Unable to load image: {img_file}")
                continue
            #resizing and normalizing data on image
            img = cv2.resize(img, (64, 64))
            img = img / 255.0 
            images.append(img)



    #lopping through to find all labels in folders
        if label_path:
                label_file = os.path.join(label_path, file_name.replace('.png', '.txt'))
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        labels.append(f.read().strip())
                else:
                    labels.append(None)

    return np.array(images), labels





#loading test and training data
print("Loading and preprocessing training data...")
train_images, train_labels = data_preprocessing(train_image_path, train_label_path)

print("Loading and preprocessing testing data...")
test_images, _ = data_preprocessing(test_image_path)


#encode labels into int
if train_labels:
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform([label for label in train_labels if label is not None])


#DATA AUGMENTATION
# Data Augmentation for training data, not validation
def data_augmentation(X_train, y_train, batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


    #train_data = datagen.flow(X_train, y_train, batch_size=batch_size)
    #return train_data
    return datagen.flow(X_train, y_train, batch_size=batch_size)



#calling functions above in order:::
#data_preprocessed = data_preprocessing(data)

print("Applying data augmentation...")
train_data = data_augmentation(train_images, train_labels)



# #saving data
print("Saving data...")
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
#np.save('test_calib.npy', test_calib)

print("Data preprocessing complete. Files saved.")