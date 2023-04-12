'''
Contains all helper and other functions used for all processes of the project
Also includes class definitions

Includes:

- batch_create(name): creating the needed folders for storage

- image_titles(dir): returns a list of all image fnames in the metadata

- generate_splits_and_files(dir, path, entries, sample_size, ratio): randomly selects
images for training/testing/validation

- class data_getter: used for management and access of image files

- process_images(fetcher, paths): used for processing the images as desired

- BatchGenerator(fetcher): generates the x_train, y_train batches as a generator

- ValBatchGenerator(fetcher): generates the x_val, y_val batches as a generator

- plot_history(history): for a given trained model, generates a visualization of its evaluation

- generate_model(generate_splits = True, model_summary = False): generates and trains a model as desired

'''
#importing the necessary libaries
import os
import numpy as np
import pandas as pd
import random
from shutil import copyfile
from PIL import Image
import matplotlib.pylab as plt

import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def batch_create(name):
  '''
  Creates a new folder in the desired google drive section with the name of the batch
  Also creates a subfolder each for train, test, valid
  Returns the path of the new folder

  Inputs:
  - name: the desired batch name

  Outputs:
  - prints a message after successfully running (for progres tracking)
  '''

  #the desired path of where the batch should be
  path = f'/content/drive/MyDrive/156_Project/gz_data/extracted_data/batch{name}/'

  #only if the batch doesn't already exist
  if(not os.path.exists(path)):

    #makes the main folder
    os.mkdir(path)
    #makes the train subfolder
    os.mkdir(f'{path}train')
    #makes the test subfolder
    os.mkdir(f'{path}test')
    #makes the validation subfolder
    os.mkdir(f'{path}valid')

  #prints a message upon successfully running the previous step (for progress tracking purposes)
  print('path: success!')

  #returning the path
  return path

def image_titles(dir):
  '''
  returns a list of all the file names in the selected folder

  inputs:
  dir - named of the directory of the desired folder

  outputs:
  prints a message for progress tracking
  returns the list of all names
  '''
  #storing all file names as a list named entries
  entries = [entry for entry in os.listdir(dir)]
  #printing a progress tracking message
  print('entries: success!')
  #returning the list entries
  return entries

def generate_splits_and_files(dir, path, entries, sample_size, ratio):
  '''
  Copies image files from all extracted files into the train, test folders of the current page
  *only if not already created*

  inputs:
  - dir: the directory of where the extracted images are
  - entries: a list of all image file titles
  - path: the path where the main batch folder is located
  - sample_size: disred sample size - number of images to be used for model generation
  - ratio: the desired train/test/valid % splits

  outputs:
  - prints a tracking message
  '''

  #reordering the image titles randomly
  shuf = np.random.permutation(entries)
  #generating the numbers to be used for train/test/valid based on the sample size and ratio
  train = int((ratio[0])*0.01*sample_size)
  test = int((ratio[1])*0.01*sample_size)
  valid = int((ratio[2])*0.01*sample_size)

  #checking to see if the train folder exists. NB: because the train/test/valid folders
  #are all created using the same func, checking for one suffices
  if os.path.exists(f'{path}train/'):

    #copying into the batch's respective folders for train/test/valid
    # entries[0:train] gives the desired train image titles (note: entries is shuffled)
    for i in range(train):
      copyfile(src = f'{dir}{shuf[i]}', dst = f'{path}train/{shuf[i]}')

    for i in range(train,(train+test)):
      # entries[train: train+test] gives the desired test image titles
      copyfile(src = f'{dir}{shuf[i]}', dst = f'{path}test/{shuf[i]}')

    for i in range((train+test), (train+test+valid)):
      # entries[train+test: train+test+valid] gives the desired valid image titles
      copyfile(src = f'{dir}{shuf[i]}', dst = f'{path}valid/{shuf[i]}')

  #progress tracking message  
  print('splits: success!')

class data_getter:    
    '''
    A class for managing file path names for train/test/valid images

    Methods:
    - get_paths: returns the titles of all images stored in a directory
    - get_all_solutions: imports the training labels as a dict
    - get_id: gets the id of an image file name after cleaning
    - find_label: returns the image label for a given image
    '''
    def __init__(self, path):    
        '''
        initialization:
        - requires a path name for where the batch is located
        - stores the train/test/val path locations
        - uses get_path to store a list of all images in train/test/val folders
        '''
        self.path = path 
        self.train_path = f'{path}train'
        self.val_path = f'{path}valid'
        self.test_path = f'{path}test'
        
        def get_paths(directory):
            '''
            returns the titles of all images stored in the dir
        
            inputs:
                -dir: the folder to be checked
            
            outputs:
                returns the list of all images in the diectory
            '''
            return [im for im in os.listdir(directory)]
        
        self.training_images_paths = get_paths(self.train_path)
        self.validation_images_paths = get_paths(self.val_path)
        self.test_images_paths = get_paths(self.test_path)    
        
        def get_all_solutions():
            '''
            imports all image labels as a dict

            inputs:
                none
            
            outputs:
                the dict of all labels
            '''
            #library required
            import csv
            #initializing the dict
            all_solutions = {}
            #google drive path where the csv of all image solutions live
            sol_doc = 'drive/MyDrive/156_Project/gz_data/training_solutions_rev1.csv' 

            #opening the csv as a reader
            with open(sol_doc, 'r') as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)
                #for each line in the csv
                for i, line in enumerate(reader):
                    all_solutions[line[0]] = [float(x) for x in line[1:]]
            #returning the dict
            return all_solutions
        
        #saving the solution dict as an object within the class
        self.all_solutions = get_all_solutions()

    def get_id(self,fname):
        '''
        gets the id of an image file name after cleaning

        inputs:
            - fname: the file name
        
        outputs:
            - the desired image file
        '''
        #getting rid of the .jpg ending and the term data in the file name
        return fname.replace(".jpg","").replace("data","")
        
    def find_label(self,val):
        '''
        returns the image label for a given image
        
        inputs:
            - val: image id
        
        outputs:
            - returns the list of the labels for the given image
        '''
        #returns the list of labels for the given image
        return self.all_solutions[val]
        
def process_images(fetcher,paths):
    '''
    Does the cropping and resizing desired - cuts by half from 424*424 to 212*212

    inputs:
        - fetcher: instance of the data_getter class
        - paths: a list of the image path titles to be resized
    
    outputs:
        - an array with all images resized and formatted
    '''
    #number of images
    count = len(paths)

    #initializing the array with 0s in the right shape -> number of images/height/width/RGB
    arr = np.zeros(shape=(count,106,106,3))
    #for each dim, half is generated by taking the center:
    #so height[0.25:0.75], width[0.25:0.75]
    left = 106
    right = 106*3
    top = 106
    bottom = 106*3
    #for each image
    for c, path in enumerate(paths):
      img = Image.open(fetcher.train_path + '/' + fetcher.training_images_paths[0])
      #croping
      img = img.crop((left,top,right,bottom))
      #resizing
      img = img.resize((106,106), Image.CUBIC)
      arr[c] = img
    #returning the array: each value of the array is a matrix of the image
    return arr

def BatchGenerator(fetcher):
    '''
    Creates a generator with X_train (images) and y_train (claassification probabilities)

    inputs:
        - fetcher: object of class data_getter
    
    outputs:
        - generator for all images in training_images_paths
    '''
    while 1:
        #for all images in the training set
        for f in fetcher.training_images_paths:
            #calling process_images to return the resized images in the fetcher object
            X_train = process_images(fetcher,[fetcher.train_path + '/' + fname for fname in [f]])
            #getting image id
            id_ = fetcher.get_id(f)
            #gfinding the label
            y_train = np.array(fetcher.find_label(id_))
            #resizing to be the desired shape
            y_train = np.reshape(y_train,(1,37))
            #yielding after each iteration
            yield (X_train, y_train)
            
def ValBatchGenerator(fetcher):
    '''
    Creates a generator with X_val (images) and y_val (claassification probabilities)

    inputs:
        - fetcher: object of class data_getter
    
    outputs:
        - generator for all images in validation_images_paths
    '''
    while 1:
        #for all images in the validation set
        for f in fetcher.validation_images_paths:
            #calling process_images to return the resized images in the fetcher object
            X_val = process_images(fetcher,[fetcher.val_path + '/' + fname for fname in [f]])
            #getting image id
            id_ = fetcher.get_id(f)
            #finding the label
            y_val = np.array(fetcher.find_label(id_))
            #resizing to be the desired shape
            y_val = np.reshape(y_val,(1,37))
            #yielding after each iteration
            yield (X_val, y_val)
                        
def plot_history(history):
    '''
    Plots two visualizations: one of the loss and the accuracy evaluations over epoch history

    inputs:
        - history: a generated model using the defined generate_model function
    '''

    #storing the training data's loss and acc using the loss function and inbuilt accuracy metric
    #storing the validation data's loss and acc using the loss function and rmse metric
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['rmse']
    val_rmse = history.history['val_rmse']
    xtick_arr = np.arange(0,len(train_loss)+1, 5)

    #ploting the training and validation loss
    fig = plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(xtick_arr)
    plt.legend(loc='best')
    plt.title('Train and Validation Loss')
    plt.show()

    print('\n')

    #plotting the training accuracy and validation rmse score
    fig = plt.figure(figsize=(10,6))
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_rmse, label='val_rmse')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(xtick_arr)
    plt.legend(loc='best')
    plt.title('Training and Validation Accuracy')
    plt.show()

def generate_model(model, opt, loss_func, name, sample_size, ratio, size_of_batch, epochs, generate_splits = True, model_summary = False, plot_vis = True):
  '''
  Generates a model with the desired configurations

  inputs:
    - model: desired model architecture
    - opt: optimization function
    - loss_func: loss function
    - name: name of the batch of data used
    - sample_size: number of the available images to be used for dataset
    - ratio: train/test/val splits
    - size_of_batch: batch size
    - epochs: number of epochs to be run
    - generate_splits: bool - True if data batch has not been created yet. False if using existing
    - model_summary: bool - True if model_summary is desired. Else, false
    - plot_viz: bool - True if visualization is desired. Else, false

  outputs:
    - prints some progress tracking messages
    - prints evaluation and progress metrics as the model is trained over each epoch
    - returns the model
  '''

  #compiles the model with the desired configuration
  model.compile(optimizer = opt, loss = loss_func, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
  
  #if model_summary is desired, prints
  if model_summary:
    model.summary()

  #directory of all images
  #note: there are 17,000 images of all 60,000 here (we werent able to extract all)
  dir = 'drive/MyDrive/156_Project/gz_data/extracted_data/images_training_rev1/'
  #storing the batch's path
  path = batch_create(name)
  #storing a list of all image fnames in the metadata
  entries = image_titles(dir)

  #if the batch's data has not been produced, then creates a new batch
  if generate_splits:
    generate_splits_and_files(dir, path, entries, sample_size, ratio)

  #instance of the data_getter class
  fetcher = data_getter(path)

  #a class used for storing and tracking the loss and accuracy data
  class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

  #for high epochs, stopping if the val_loss doesnt change without 50 epochs
  early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
  #instance of the LossHistory class
  history = LossHistory()
  #checkpointing
  checkpointer = ModelCheckpoint(filepath='tmp/weights.hdf5', verbose=1, save_best_only=True)

  #batch size of each epochs
  batch_size = size_of_batch
  #number of batches for training and validation
  steps_to_take = int(len(fetcher.training_images_paths)/batch_size)
  val_steps_to_take = int(len(fetcher.validation_images_paths)/batch_size)
                  #typically be equal to the number of unique samples if your dataset
                  #divided by the batch size.

  #training the model
  hist = model.fit_generator(BatchGenerator(fetcher),
                      steps_per_epoch=steps_to_take, 
                      epochs=epochs,
                      validation_data=ValBatchGenerator(fetcher),
                      validation_steps=val_steps_to_take,
                      verbose=2,
                      callbacks=[history,checkpointer,early_stopping],
                   )
  
  #if a visualization is desired
  if plot_vis:
     plot_history(hist)
  
  #returning the model
  return hist