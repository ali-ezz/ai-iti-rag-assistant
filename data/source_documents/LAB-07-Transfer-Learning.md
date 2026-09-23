# LAB-07-Transfer-Learning

Source notebook: LAB-07-Transfer-Learning.ipynb

<center><a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a></center>

# 7. Assessment

Congratulations on going through today's course! Hopefully, you've learned some valuable skills along the way and had fun doing it. Now it's time to put those skills to the test. In this assessment, you will train a new model that is able to recognize fresh and rotten fruit. You will need to get the model to a validation accuracy of `92%` in order to pass the assessment, though we challenge you to do even better if you can. You will have the use the skills that you learned in the previous exercises. Specifically, we suggest using some combination of transfer learning, data augmentation, and fine tuning. Once you have trained the model to be at least 92% accurate on the validation dataset, save your model, and then assess its accuracy. Let's get started!

## 7.1 The Dataset

In this exercise, you will train a model to recognize fresh and rotten fruits. The dataset comes from [Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification), a great place to go if you're interested in starting a project after this class. The dataset structure is in the `data/fruits` folder. There are 6 categories of fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. This will mean that your model will require an output layer of 6 neurons to do the categorization successfully. You'll also need to compile the model with `categorical_crossentropy`, as we have more than two categories.

<img src="./images/fruits.png" style="width: 600px;">

## 7.2 Load ImageNet Base Model

We encourage you to start with a model pretrained on ImageNet. Load the model with the correct weights. Because these pictures are in color, there will be three channels for red, green, and blue. We've filled in the input shape for you. If you need a reference for setting up the pretrained model, please take a look at [notebook 05b](05b_presidential_doggy_door.ipynb) where we implemented transfer learning.

## 7.3 Freeze Base Model

Next, we suggest freezing the base model, as done in [notebook 05b](05b_presidential_doggy_door.ipynb). This is done so that all the learning from the ImageNet dataset does not get destroyed in the initial training.

## 7.4 Add Layers to Model

Now it's time to add layers to the pretrained model. [Notebook 05b](05b_presidential_doggy_door.ipynb) can be used as a guide. Pay close attention to the last dense layer and make sure it has the correct number of neurons to classify the different types of fruit.

The later layers of a model become more specific to the data the model trained on. Since we want the more general learnings from VGG, we can select parts of it, like so:

Once we've taken what we've wanted from VGG16, we can then add our own modifications. No matter what additional modules we add, we still need to end with one value for each output.

## 7.5 Compile Model

Now it's time to compile the model with loss and metrics options. We have 6 classes, so which loss function should we use?

## 7.6 Data Transforms

To preprocess our input images, we will use the transforms included with the VGG16 weights.

Try to randomly augment the data to improve the dataset. Feel free to look at [notebook 04a](04a_asl_augmentation.ipynb) and [notebook 05b](05b_presidential_doggy_door.ipynb) for augmentation examples. There is also documentation for the [TorchVision Transforms class](https://pytorch.org/vision/stable/transforms.html).

**Hint**: Remember not to make the data augmentation too extreme.

## 7.7 Load Dataset

Now it's time to load the train and validation datasets.

Select the batch size `n` and set `shuffle` either to `True` or `False` depending on if we are `train`ing or `valid`ating. For a reference, check out [notebook 05b](05b_presidential_doggy_door.ipynb).

## 7.8 Train the Model

Time to train the model! We've moved the `train` and `validate` functions to our [utils.py](./utils.py) file. Before running the below, make sure all your variables are correctly defined.

It may help to rerun this cell or change the number of `epochs`.

## 7.9 Unfreeze Model for Fine Tuning

If you have reached 92% validation accuracy already, this next step is optional. If not, we suggest fine tuning the model with a very low learning rate.

## 7.10 Evaluate the Model

Hopefully, you now have a model that has a validation accuracy of 92% or higher. If not, you may want to go back and either run more epochs of training, or adjust your data augmentation. 

Once you are satisfied with the validation accuracy, evaluate the model by executing the following cell. The evaluate function will return a tuple, where the first value is your loss, and the second value is your accuracy. To pass, the model will need have an accuracy value of `92% or higher`.

## 7.11 Run the Assessment

To assess your model run the following two cells.

**NOTE:** `run_assessment` assumes your model is named `my_model`. If for any reason you have modified these variable names, please update the names of the arguments passed to `run_assessment`.

## 7.12 Generate a Certificate

If you passed the assessment, please return to the course page (shown below) and click the "ASSESS TASK" button, which will generate your certificate for the course.

<img src="./images/assess_task.png" style="width: 800px;">

<center><a href="https://www.nvidia.com/dli"> <img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/> </a></center>
