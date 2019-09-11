from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type = str, default = "knn", help = "type of python machine learning to use")
ap.add_argument("-d", "--dataset", type = str, default = "Original Data 3 Kelas", help = "path to directory containing the '3scenes' dataset")
args = vars(ap.parse_args())

def extract_color_stats(image):
	# Split the input image into its respective RGB colour channels and then create a feature vector with six values; the mean and std for each of the three channels
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
	# Return the set of features
	return features

# Define the dictionary of models our script can use
models = {
	"knn": KNeighborsClassifier(n_neighbors = 1),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver = "lbfgs", multi_class = "auto"),
	"svm": SVC(kernel = "rbf", gamma = "auto"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators = 100),
	"mlp": MLPClassifier(),
	"perceptron": Perceptron(max_iter = 50)
}

# Grab all image paths in the input dataset directory, initialise our list of extracted features and corresponding labels
print("[INFO] extracting image features {}".format(args["dataset"]))
imagePaths = paths.list_images(args["dataset"])
data =[]
labels = []

# Loop over input images
for imagePath in imagePaths:
	# Load the input image from disk, compute colour channel statistics, and then update the data list
	image = Image.open(imagePath)
	features = extract_color_stats(image)
	data.append(features)

	# Extract the class label from the file path and update the labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Encode labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Perform a training and testing split, using 75% of data for training and 25% of data for testing
(trainx, testx, trainy, testy) = train_test_split(data, labels, test_size = 0.1)
#(trainx, valx, trainy, valy) = train_test_split(trainx, trainx, test_size=0.1, random_state=1)

# Train the model
print("[INFO] using {} model".format(args["model"]))
model = models[args["model"]]
model.fit(trainx, trainy)

# Make predictions, show classification report
print("[INFO] evaluating...")
predictions = model.predict(testx)
print(classification_report(testy, predictions, target_names = le.classes_))