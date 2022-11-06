# Age prediction Deep learning project

## Aim of the project:
The aim of the project was to crete a program to predict a human age.

## Process:
1. Getting the data
- Checking the webpages that contain both images and corresponding ages. Winner IMDB.com
- IMDB also had a *.txt dataset that contained a person ID and their age.
- Splitting the ages into categories and analysing number of potential samples in each class.
- Parsing webpages of each age class to get a link to the image & download. (Requests, RequestsTor, BeautifulSoup)
- There were downloaded 11_000 distinct images.

2. Data cleaning, Data transformation
- Removing manually all images that do not match the requirements
- Extraction of faces from raw images by opencv2
- Changing brightness of images
- Conversion to grayscale
- Extraction of labels of each class
- Conversion of images to numpy arrays
- After data augmentation, count was increased to 36_000 - 51_000 images.

3. Training & evaluating model
- Training Convolutional NN
- Training fully connected NN
- Hyperparameter tuning
- Going back to step 2. to augment the data and repeating the training.

4. GUI
- Created GUI in Tkinter with visualised predictions for each image.

## Limitation:
- Ages and appearances of actors might not reflect general population
- Ages of people might not have 100% reflected the age on the shown photos on IMDB.
- Not enough data
- Hardware (RTX 2080S 8GB)
- Data cleaning of extracted faces by opencv2 was not done

## Status of the project:
Functional

## Results / Benchmarks:
Benchmark of age prediction on the same data.

Random - 16% \
Model - 30% (Recall, Precision, Accuracy) \
Human - 50%

![Image](github_img.jpg)