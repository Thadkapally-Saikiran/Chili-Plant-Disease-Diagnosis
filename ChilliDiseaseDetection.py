# Import necessary modules and libraries for creating GUI, data manipulation, plotting, image processing, machine learning, and file operations
from tkinter import messagebox  # Import messagebox module from tkinter for displaying messages in GUI applications
from tkinter import *  # Import tkinter library for creating GUI elements
from tkinter import simpledialog  # Import simpledialog module from tkinter for creating simple dialog boxes
import tkinter  # Import tkinter library for GUI functionality
import matplotlib.pyplot as plt  # Import pyplot module from matplotlib for plotting graphs and charts
import numpy as np  # Import numpy library for numerical computing and array manipulation
from tkinter import ttk  # Import ttk module from tkinter for themed widgets
from tkinter import filedialog  # Import filedialog module from tkinter for opening file dialogs in GUI applications
from keras.utils import to_categorical  # Import to_categorical function from keras.utils for one-hot encoding labels
from keras.models import Sequential  # Import Sequential class from keras.models for creating a sequential neural network model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten  # Import necessary layers from tensorflow.keras.layers for building the neural network
from sklearn.metrics import accuracy_score  # Import accuracy_score function from sklearn.metrics for evaluating model accuracy
import os  # Import os module for interacting with the operating system
import cv2  # Import OpenCV library for computer vision tasks and image processing
from keras.layers import Convolution2D  # Import Convolution2D layer from keras.layers for 2D convolution on images
from keras.layers import MaxPooling2D  # Import MaxPooling2D layer from keras.layers for 2D max pooling on images
import pickle  # Import pickle module for serializing and deserializing Python objects
from keras.models import model_from_json  # Import model_from_json function from keras.models for loading a model architecture from JSON format


# Create the main application window using Tkinter
main = Tk()

# Set the title of the window to indicate the purpose of the application
main.title("Disease Detection in Chilli plants and Remote Monitoring of agricultural parameters")

# Set the initial dimensions of the window (width: 1300 pixels, height: 1200 pixels)
main.geometry("1300x1200")

# Declare global variables to be used across functions: filename, image data (X), labels (Y), trained model, and accuracy information
global filename
global X, Y
global model
global accuracy

# Define a list of disease labels for prediction
plants = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']


# Define a function named "uploadDataset" that will be called when the corresponding button is clicked in the GUI
def uploadDataset():
    # Declare X and Y as global variables to be accessed and modified within this function
    global X, Y
    
    # Declare filename as a global variable to store the path of the selected directory
    global filename
    
    # Clear the text widget content from the first character to the last (1.0 to END)
    text.delete('1.0', END)
    
    # Open a file dialog to select a directory and store its path in the 'filename' variable
    filename = filedialog.askdirectory(initialdir=".")
    
    # Display a message in the text widget indicating that the dataset has been loaded
    text.insert(END, 'dataset loaded\n')


# Define a function named "imageProcessing" that will be executed when the corresponding button is clicked in the GUI
def imageProcessing():
    # Clear the text widget content from the first character to the last (1.0 to END)
    text.delete('1.0', END)
    
    # Declare X and Y as global variables to be accessed and modified within this function
    global X, Y
    
    # Load preprocessed image data and labels from saved files using NumPy
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    
    # Retrieve and reshape an example image from the loaded dataset (X[20]) to (64x64x3) dimensions
    img = X[20].reshape(64, 64, 3)
    
    # Perform one-hot encoding on the labels (Y) using the to_categorical function from Keras utils
    Y = to_categorical(Y)
    
    # Convert loaded image data and labels to NumPy arrays for further processing
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Convert image data to 32-bit floating-point format
    X = X.astype('float32')
    
    # Normalize pixel values of images to be between 0 and 1 (scaling)
    X = X / 255
    
    # Generate an array of indices from 0 to the number of images and shuffle them randomly
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Shuffle the images (X) and labels (Y) based on the generated random indices
    X = X[indices]
    Y = Y[indices]
    
    # Display a message in the text widget indicating that image processing is completed
    text.insert(END, 'image processing completed\n')
    
    # Display the example image in a window using OpenCV (cv2) after resizing it to (250x250) dimensions
    cv2.imshow('ff', cv2.resize(img, (250, 250)))
    
    # Wait until a key is pressed in the image window (0 parameter means infinite waiting time)
    cv2.waitKey(0)


# Define a function named "cnnModel" to load a pre-trained CNN model and its weights
def cnnModel():
    # Declare model and accuracy as global variables to be accessed and modified within this function
    global model
    global accuracy
    
    # Clear the text widget content from the first character to the last (1.0 to END)
    text.delete('1.0', END)
    
    # Check if the model architecture file 'model/model.json' exists in the specified path
    if os.path.exists('model/model.json'):
        # Open and read the model architecture file in JSON format
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            # Load the model architecture from the JSON file using model_from_json function
            model = model_from_json(loaded_model_json)
        # Close the opened JSON file
        json_file.close()
        
        # Load the pre-trained model weights from the file 'model/model_weights.h5'
        model.load_weights("model/model_weights.h5")
        
        # Print the summary of the loaded CNN model architecture
        print(model.summary())
        
        # Open the history file containing accuracy data and load it using pickle
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        
        # Retrieve accuracy values from the loaded history data and calculate the accuracy at the 10th epoch
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        
        # Display the CNN model's prediction accuracy in the text widget
        text.insert(END, "CNN Chilli Disease Detection Model Prediction Accuracy = " + str(acc))

    else:
        # Define a Sequential model, which allows linearly stacking layers on top of each other
        model = Sequential()
        
        # Add a 2D convolutional layer with 32 filters, each of size (3x3), and ReLU activation function
        # Input shape is set to (64, 64, 3) representing 64x64 images with 3 RGB color channels
        model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        
        # Add a 2D max-pooling layer with pool size (2x2) to downsample the spatial dimensions
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Add another 2D convolutional layer with 32 filters, each of size (3x3), and ReLU activation function
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        
        # Add another 2D max-pooling layer with pool size (2x2) to downsample the spatial dimensions further
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten the 3D output to 1D array to prepare for fully connected layers
        model.add(Flatten())
        
        # Add a dense (fully connected) layer with 256 neurons and ReLU activation function
        model.add(Dense(units=256, activation='relu'))
        
        # Add the output layer with 2 neurons and softmax activation for binary classification (healthy or disease)
        model.add(Dense(units=2, activation='softmax'))
        
        # Compile the model using Adam optimizer, categorical crossentropy loss, and accuracy as a metric
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Print a summary of the model architecture, showing layers, output shapes, and parameters
        print(model.summary())

        # Train the CNN model using the given image data (X) and labels (Y) with specified parameters
        # 'batch_size' is the number of samples used in each iteration for gradient descent
        # 'epochs' is the number of times the entire dataset is passed forward and backward through the neural network
        # 'validation_split' is the fraction of the training data to be used as validation data during training (20% in this case)
        # 'shuffle' parameter shuffles the training data before each epoch
        # 'verbose' parameter controls the amount of logging during training (verbosity level 2 in this case)
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)

        # Save the trained model's weights to a file for future use
        model.save_weights('model/model_weights.h5')

        # Convert the model to JSON format and save it to a file for future use
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)

        # Close the JSON file after writing the model architecture
        json_file.close()

        # Open a binary file in write mode to store the training history (accuracy and loss) using pickle
        f = open('model/history.pckl', 'wb')

        # Dump the history dictionary containing accuracy and loss to the opened binary file
        pickle.dump(hist.history, f)

        # Close the binary file after storing the history
        f.close()

        # Open the stored history file in read mode using pickle
        f = open('model/history.pckl', 'rb')

        # Load the accuracy and loss values from the stored history
        accuracy = pickle.load(f)

        # Close the history file after loading the data
        f.close()

        # Extract the accuracy values from the loaded history dictionary
        acc = accuracy['accuracy']

        # Get the accuracy value corresponding to the 10th epoch and multiply it by 100 for percentage accuracy
        acc = acc[9] * 100

        # Display the CNN model's prediction accuracy in the text widget of the GUI
        text.insert(END, "CNN Chilli Disease Detection Model Prediction Accuracy = " + str(acc))

        
# Define a function named "predict" that will be executed when the corresponding button is clicked in the GUI
def predict():
    # Declare the 'model' variable as global to access the pre-trained CNN model within this function
    global model
    
    # Open a file dialog to select an image file from the 'testImages' directory
    filename = filedialog.askopenfilename(initialdir="testImages")
    
    # Read the selected image using OpenCV and resize it to (64x64) dimensions
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))
    
    # Convert the resized image to a NumPy array and reshape it to match the input shape of the model
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    
    # Convert the image array to 32-bit floating-point format and normalize pixel values to be between 0 and 1
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test / 255
    
    # Make predictions using the pre-trained model on the processed test image
    preds = model.predict(test)
    
    # Get the index of the class with the highest prediction probability
    predict = np.argmax(preds)
    
    # Read the original image again for display purposes and resize it to (800x400) dimensions
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))
    
    # Add a text label to the image indicating the recognized disease class based on the prediction
    cv2.putText(img, 'Chilli Disease Recognized as: ' + plants[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the image with the prediction label in a window using OpenCV
    cv2.imshow('Chilli Disease Recognized as: ' + plants[predict], img)
    
    # Wait until a key is pressed in the image window (0 parameter means infinite waiting time)
    cv2.waitKey(0)


# Define a function named "graph" responsible for plotting and displaying accuracy and loss graphs
def graph():
    # Retrieve accuracy and loss values from the 'accuracy' dictionary (global variable)
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    
    # Create a new figure for the plot with a specific size (10x6 inches)
    plt.figure(figsize=(10, 6))
    
    # Enable gridlines on the plot
    plt.grid(True)
    
    # Set labels for the x and y axes of the plot
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    
    # Plot accuracy values as red circles connected by lines ('ro-' represents red circles)
    plt.plot(acc, 'ro-', color='green')
    
    # Plot loss values as blue circles connected by lines ('ro-' represents blue circles)
    plt.plot(loss, 'ro-', color='blue')
    
    # Add legends to the plot to indicate which line corresponds to accuracy and which one corresponds to loss
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    
    # Set the title of the plot
    plt.title('Iteration Wise Accuracy & Loss Graph')
    
    # Display the generated plot to visualize the accuracy and loss trends over iterations
    plt.show()

    
# Define a function named "close" that will be executed when the corresponding button is clicked in the GUI
def close():
    # Destroy the main window, closing the application
    main.destroy()
    
    # Clear the text widget content from the first character to the last (1.0 to END)
    text.delete('1.0', END)

    
# Define a font style named "font" with 'times' font family, font size 15, and bold weight
font = ('times', 15, 'bold')

# Create a Label widget named "title" within the main Tkinter window (main)
# Set the text of the label to the specified string describing the application purpose
title = Label(main, text='Disease Detection in Chilli plants and Remote Monitoring of agricultural parameters')

# Configure the font style of the label to the previously defined "font" style
title.config(font=font)

# Set the height and width of the label widget to 3 and 120 respectively
title.config(height=3, width=120)

# Place the label widget at the specified coordinates (x=0, y=5) within the main window
title.place(x=0, y=5)

# Define a font style named "font1" with the 'times' font family, size 13, and bold weight
font1 = ('times', 13, 'bold')

# Define another font style named "ff" with the 'times' font family, size 12, and bold weight
ff = ('times', 12, 'bold')


# Create a button widget with the specified text and bind the "uploadDataset" function to it
uploadButton = Button(main, text="Upload Chilli Plant Images Dataset", command=uploadDataset)
# Set the position of the upload button within the main window using the place geometry manager
uploadButton.place(x=20, y=100)
# Configure the font style of the upload button
uploadButton.config(font=ff)

# Create a button widget for image processing and bind the "imageProcessing" function to it
processButton = Button(main, text="Image Processing & Normalization", command=imageProcessing)
# Set the position of the image processing button within the main window
processButton.place(x=20, y=150)
# Configure the font style of the image processing button
processButton.config(font=ff)

# Create a button widget for displaying the trained CNN model and bind the "cnnModel" function to it
modelButton = Button(main, text="Trained CNN Model", command=cnnModel)
# Set the position of the model button within the main window
modelButton.place(x=20, y=200)
# Configure the font style of the model button
modelButton.config(font=ff)

# Create a button widget for uploading a test image and predicting disease, bind the "predict" function to it
predictButton = Button(main, text="Upload Test Image & Predict Disease", command=predict)
# Set the position of the predict button within the main window
predictButton.place(x=20, y=250)
# Configure the font style of the predict button
predictButton.config(font=ff)

# Create a button widget for displaying the accuracy and loss graph, bind the "graph" function to it
graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
# Set the position of the graph button within the main window
graphButton.place(x=20, y=300)
# Configure the font style of the graph button
graphButton.config(font=ff)

# Create a button widget for exiting the application, bind the "close" function to it
exitButton = Button(main, text="Exit", command=close)
# Set the position of the exit button within the main window
exitButton.place(x=20, y=350)
# Configure the font style of the exit button
exitButton.config(font=ff)


# Define a font style named "font1" with 'times' font family, 12-point size, and bold weight
font1 = ('times', 12, 'bold')
# Create a Text widget named "text" with a height of 30 lines and width of 85 characters
text = Text(main, height=30, width=85)
# Create a Scrollbar widget named "scroll" associated with the Text widget for vertical scrolling
scroll = Scrollbar(text)
# Configure the Text widget to use the Scrollbar for vertical scrolling
text.configure(yscrollcommand=scroll.set)
# Place the Text widget on the main GUI window at coordinates (410, 100)
text.place(x=410, y=100)
# Configure the font style of the Text widget to be 'font1'
text.config(font=font1)


# Set the background color of the main application window to violet
main.config(bg="violet")
# Start the Tkinter main event loop, allowing the GUI to respond to user interactions
main.mainloop()
