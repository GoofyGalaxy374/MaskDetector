# MaskDetector
 ## The goal of this project is to train a Convolutional Neural Network (CNN) to successfully recognize when a person is wearing a mask, and when not.



# Dataset
### The dataset is split into **2 categories**, which the model uses for classification:
<ul>
    <li>With mask</li>
    <li>Without mask</li>
</ul>

### They both contain approximately 5000 images of people with masks and people without masks

# Architecture/Structure of the Convolutional Neural Network
## The structure of the CNN is fairly simple. It consists of the following
<ul>
    <li>3 convolutional layers, followed by 3 max pooling layers. </li>
    <li>2 dense, final layers.</li>
</ul>

## **Input**

###  The input that the CNN uses, comes from the camera. It is then reshaped to 256px x 256px, turned into a matrix and it is finally turned to a single color channel - grayscale. 
### The reason that it is turned into grayscale is that with grayscale, the color channgel is single, thus we have to only train the network to see certain shades(pixel values) at certain positions on the input.

## **Regularizations**
### After the input is set up to be accepted from the model, it is then regularized - the values that come from the input are put in between -1 and 1. 

## **Feeding the data to the network**
### Finally, after all the preprocessing and regularuzations, the data is sent to the model which makes predictions upon it. The predictions are always numbers between -1 and 1, where -1 means an absolute certainty that the person isn't wearing a mask, and 1 means the opposite.

## **Visualization**
### The model's predictions are visible in real time on the user's screen, thanks to the openCV2 library.

# Guidelines
## **Prerequisites**
<ul>
<li>
    <strong> Python 3.9 +</strong>
</li>
<li>
    <strong> Tensorflow </strong>
</li>
<li>
    <strong> Numpy </strong>
</li>
</ul>


## **Installation and execution of the program**
<ul>
    <li> Download the project from the Github Repository </li>
    <li> Extract the data from the archives into a folder</li>
    <li> Open the terminal/CMD(also known as console) in the folder, where you extracted the project files</li>
    <li> Run the following command <code> python camera_module.py </code> </li>
</ul>

## Sources
### The dataset utilized **is not mine**. You can access it from this Github Repo - https://github.com/prajnasb/observations/tree/master/experiements/data
