import torch
import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision

import constants
from sklearn.metrics import confusion_matrix
from torch import nn, optim

classes = [constants.Masked, constants.UnMasked, constants.NonPerson]


def generateConfusionMatrix(test_data, test_prediction):
    print("Confusion Matrix:\n", confusion_matrix(test_data, test_prediction))


def generateAccuracyResult(test_data, cnn):
    correct = 0
    total = 0
    test_data_size = len(test_data)
    with torch.no_grad():
        for images, labels in test_data:
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * (correct / total)
    return accuracy


def generatePrecisionResult(test_data, test_prediction):
    from sklearn.metrics import precision_score
    precision = precision_score(test_data, test_prediction, average=None)
    return precision


def generateRecallResult(test_data, test_prediction):
    from sklearn.metrics import recall_score
    recall = recall_score(test_data, test_prediction, average=None)
    return recall


def generateF1MeasureResult(precision, recall):
    f1measure = 2 * (precision * recall) / (precision + recall)
    return f1measure


class ImageClassifier():
    ImageDimensions = constants.imageSize
    trained_Data = []
    Labels = {constants.NonPerson: 2, constants.Masked: 1, constants.UnMasked: 0}

    def ImageDataSetPrep(self):
        path = constants.training_dataSavedPath + constants.trained_datasetName
        base_Path = Path(constants.Base_Path)
        print('path', constants.Base_Path)
        masked_Path = base_Path / constants.Masked_Folder
        unmasked_Path = base_Path / constants.UnMasked_Folder
        nonPerson_Path = base_Path / constants.NonPerson_Folder
        path_dirs = [[nonPerson_Path, 2], [masked_Path, 1], [unmasked_Path, 0]]
        count = 0
        if not os.path.exists(base_Path):
            raise Exception("The data path doesn't exist")
        for Dir_path, label in path_dirs:
            print("Current Label is ", label)
            for image in os.listdir(Dir_path):
                imagePath = os.path.join(Dir_path, image)
                try:
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (self.ImageDimensions, self.ImageDimensions))
                    print(constants.RESIZED + "\image" + str(count) + ".jpg")
                    if label == 2:
                        cv2.imwrite(constants.RESIZED + "\class2\image" + str(count) + ".jpg", img)
                    if label == 1:
                        cv2.imwrite(constants.RESIZED + "\class1\image" + str(count) + ".jpg", img)
                    if label == 0:
                        cv2.imwrite(constants.RESIZED + "\class0\image" + str(count) + ".jpg", img)
                    count += 1
                except Exception as e:
                    print('Error Occured while processing images')
                    print(e)
                    pass


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(160000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3))

    def forward(self, x):
        return self.cnn_layers(x)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.ImageFolder(root=constants.RESIZED, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
    testset = torchvision.datasets.ImageFolder(root=constants.TEST, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
    net = Net()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 1 test images: %d %%' % (
            100 * correct / total))

