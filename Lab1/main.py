import kagglehub
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import glob
import random

projectDataPath = '../content/dataset'
dataset_dir = os.path.join(projectDataPath, '1/Data/')
train_dir = os.path.join(dataset_dir, 'Train_Data')
test_dir = os.path.join(dataset_dir, 'Test_Data')

#download the data and move
if not (os.path.exists(projectDataPath)):
    os.makedirs(projectDataPath, exist_ok=True)

    downloadPath = kagglehub.dataset_download("mohnishsaiprasad/forest-fire-images")

    shutil.move(downloadPath, projectDataPath)

#read filenames of all images
train_data_images = {
    "Fire":  glob.glob(os.path.join(train_dir, 'Fire/*.jpg')),
    "Non_Fire": glob.glob(os.path.join(train_dir, 'Non_Fire/*.jpg'))
}
test_data_images = {
    "Fire":glob.glob(os.path.join(test_dir, 'Fire/*.jpg')),
    "Non_Fire": glob.glob(os.path.join(test_dir, 'Non_Fire/*.jpg'))
}

X, y, X_test, y_test = [], [], [], []
#where X is the matrix of training data and _test is the test data
#where y is the label

errCount = 0


def readAndLabelImage(targetImageArray, targetLabelArray, image, label):
    try:
        targetImageArray.append(Image.open(image))
        if(label == 'Fire'):
            intLabel = 1
        elif(label == 'Non_Fire'):
            intLabel = 0
        else:
            raise Exception("Incorrect label provided")
        targetLabelArray.append(intLabel)
    except Exception as e:
        return 1
    return 0

#sort into seperate arrays X, y, X_test, y_test
for fire_image in train_data_images['Fire']:
    errCount += readAndLabelImage(X, y, fire_image, 'Fire')

for non_fire_image in train_data_images['Non_Fire']:
    errCount += readAndLabelImage(X, y, non_fire_image, 'Non_Fire')

for fire_image in test_data_images['Fire']:
    errCount += readAndLabelImage(X_test, y_test, fire_image, 'Fire')

for non_fire_image in test_data_images['Non_Fire']:
    errCount += readAndLabelImage(X_test, y_test, non_fire_image, 'Non_Fire')

print('Error Count: ', errCount)


def testRandom16():
    total_images = 16
    combined_list = list(zip(X, y))
    Samples = random.sample(combined_list, 16)
    grid_size = (4, 4)
    fig, axes = plt.subplots(grid_size[0], grid_size[1])
    axes = axes.flatten()
    for idx in range(total_images):
        ax = axes[idx]
        ax.axis('off')

        img, label = Samples[idx]

        ax.imshow(img)
        ax.set_title(label, fontsize=10)
    plt.tight_layout()
    plt.show()

#testRandom16()

def checkArrayLengths():
    print(len(X))
    print(len(y))
    print(len(X_test))
    print(len(y_test))

checkArrayLengths()

#not needed - just an example
def printExampleFileNames():
    print('Train fire: ', train_data_images['Fire'][0])
    print('Train non_fire: ', train_data_images['Non_Fire'][0])
    print('Test fire: ', test_data_images['Fire'][0])
    print('Test non_fire: ', test_data_images['Non_Fire'][0])

#printExampleFileNames()

#not needed - just an example
def showExampleImages(pathToDataset):

    fire_sample_image = Image.open(pathToDataset+'/1/Data/Train_Data/Fire/F_0.jpg')
    non_fire_sample_image = Image.open(pathToDataset+'/1/Data/Train_Data/Non_Fire/NF_10.jpg')

    fig, axes = plt.subplots(1, 2)
    axes = axes.flatten()

    axes[0].imshow(fire_sample_image)
    axes[0].set_title('Fire Sample', fontsize=10)

    axes[1].imshow(non_fire_sample_image)
    axes[1].set_title('Non Fire Sample', fontsize=10)

    plt.show()


