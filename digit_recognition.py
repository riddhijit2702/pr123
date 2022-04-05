#Pair programming is an important technique for developing higher quality code, faster while also reducing risk of errors.
#Importing all the important models and install them if not installed on your device
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Google Colab is available on the browser. 
# When we go to any link, many of the links start with an https:// instead of an http:// 
# The s in the https:// means that we are trying to establish a secure connection. 
# Browser provides us with SSL when we try to open a link.
# Here since we are running a python script locally on our machine, the openML thinks that our python script cannot be trusted since there is no SSL, 
# We import a library known as ssl. 

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context
#We first have an if condition where we are checking if our python environment’s “PYTHONHTTPSVERIFY” is an empty string. 
#This means that our Python script is not HTTPS Verified.
#Next, we are seeing if our ssl’s unverified context is created or not. 
#Now, if our script does not have https verification 
# and if it also does not have an ssl’s unverified context, 
# then we are creating a default https unverified context for our python script using the SSL module. 



#Fetching the data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  # try block lets us test the code for errors
  # except block lets us handle the errors
  try:
    ret, frame = cap.read()

    # As we are going to be giving our model direct input from a video we don't want it to get confused due to all the colors 
    # so we'll set the color of the video to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    #This rectangle will be the region of interest. 
    #This rectangle will serve the purpose as it will be the only place area where the model will detect the digit.  
    height, width = gray.shape
    #shape- It returns a tuple of number of rows, columns and channels  
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to pil format
    #Image.fromarray()- Creates an image memory from an object exporting the array interface 
    #The Image module provides a class with the same name which is used to represent a PIL image.
    im_pil = Image.fromarray(roi)

    # convert to grayscale image
    # 'L' format means each pixel is represented by a single value from 0 to 255
    #The L means that is just stores the Luminance. 
    # It is very compact, but only stores a greyscale, not colour.
    image_bw = im_pil.convert('L')
    #Resize the image
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    #Antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution.

    #Inverting the image as camera captures inverted image
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

    #After inverting the image we also have to make it scalar to get the minimum pixel 
    #and limit its value between 0 and 255 and then getting the maximum pixel of the image. 
    pixel_filter = 20
    #percentile function converts the values in scalar quantity 
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    #using clip to limit the values between 0,255
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    #max function get the maximum of the given numbers
    max_pixel = np.max(image_bw_resized_inverted)
     #converting into an array
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    
    #creating a test sample and making a prediction
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    
  except Exception as e:
    pass
