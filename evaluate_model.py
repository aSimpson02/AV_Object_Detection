#Importing relevant libraries:::
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.utils import to_categorical


#model = load_model("cnn_model.h5")
model = load_model("my_model.keras")

#finding paths to test data
test_images_path = "test_images.npy"
test_labels_path = "test_labels.npy"


#loading test data
test_images = np.load(test_images_path)
test_labels = np.load(test_labels_path)

#one-hot encodings on test labels [converting from string to numeric values so it can be understood.]
test_labels = to_categorical(test_labels, num_classes=10)



#Evalute and Iterate over model:::
def evaluate_model(model, test_images, test_labels):

    #acuracy/loss::
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f"test loss: {loss}")
    print(f"test accuracy: {accuracy}")

    #predictions::
    predictions = model.predict(test_images)
    predicted = np.argmax(predictions, axis=1)
    true = np.argmax(test_labels, axis=1)

    #classification
    print("classification Report:")
    print(classification_report(true, predicted))

    #confusion matrix??


#matplotlib to plot data???


#calling to evaluate [+ plot - later]
evaluate_model(model, test_images, test_labels)
#plot_training_history(model_train)