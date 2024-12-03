#Importing relevant libraries:::
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



#trainig paths for processed and augmented data
#these steps are found in the load_data.py file
train_images_path = "train_images.npy"
train_labels_path = "train_labels.npy"

train_images = np.load(train_images_path, allow_pickle=True)
train_labels = np.load(train_labels_path, allow_pickle=True)


#checkng data
#making sure first 10 outputs are an array
print(train_labels[:10])  
print(type(train_labels)) 
print(train_labels.shape)


# Define class mapping (adjust as needed)
class_mapping = {
    'car': 0,
    'irrelevant': 1,
    'pedestrian': 2,
    'bike': 3,
    'lorry': 4,
    'bus': 5,
    'traffic Light': 6,
    'road Sign': 7,
    'motorbike': 8,
}


#strings to numeric values
train_labels = [
    class_mapping.get(label.split()[0], -1) if label is not None else -1
    for label in train_labels
]
#train_labels = [class_mapping.get(label.split()[0], -1) if label is not None else -1 for label in train_labels]

print(train_labels[:10]) 



#checking for anybinvalid labels
if -1 in train_labels:
    print("Warning: Some labels not mapped to class ID.")


train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels, num_classes=len(class_mapping))


train_labels = np.array([0 if label is None else int(label) for label in train_labels])

#one-hot encodings on train labels [converting from string to numeric values so it can be understood.]
#train_labels = to_categorical(train_labels, num_classes=10)
#train_labels = to_categorical(train_labels, num_classes=len(class_mapping))


print(train_labels[:10])






#split data for training and validation on the models
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

#normalising images
X_train = X_train / 255.0
X_val = X_val / 255.0

#CNN Model Architetcure:::
    #function that contains architecture
def building_model():
    #model = sequential
    model = Sequential()

    
    #crop any unecessary parts of images 
    # normalise ?

    #5x5 layers - depth = (24, 36, 48)
    model.add(Conv2D(24, (5, 5), activation='elu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(36, (5, 5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))


    #3x3 layers - depth = (64, 64)

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))


    #for flattening + connecting layers:
    #flatter - dense - dropout - dense - dense - dense
    #use elu activation instead of relu activation
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(10, activation='softmax'))


    #return the model for training 
    return model


#making model globally accessible
model = building_model()
model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

#Training the Model:::
#functions to train the model
#use ADAM optimizer for training

def training_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size = 32):
    updated_model = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    return updated_model


#save model for evaluation:
#model.save("model.h5") ???
model_train = training_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
model.save('my_model.keras')
#model.save("cnn_model.h5")