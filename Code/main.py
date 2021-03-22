import torch
import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision

from sklearn.metrics import confusion_matrix
from torch import nn, optim




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


def generateF1MeasureResult(test_data, test_prediction):
    from sklearn.metrics import f1_score
    f1measure = f1_score(test_data, test_prediction, average=None)
    return f1measure




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
    trainset = torchvision.datasets.ImageFolder(root="../Images/resized", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
    testset = torchvision.datasets.ImageFolder(root="../Images/test", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
    net = Net()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    """
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    """
    # Save
    PATH = "state_dict_model.pt"
    net = torch.load(PATH)
    #torch.save(net, PATH)
    
    pred_data=[]
    test_data=[]
    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = net(images)
        test_data.append(labels.numpy().tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        pred_data.append(predicted.numpy().tolist())
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / total))
    
    pred_data =pred_data[0]
    test_data = test_data[0]
    print(len(test_data)," and ",len(pred_data))
    print("TST DATA" ,test_data,"PRED DATA::",pred_data)
    print("confusion Matrix",generateConfusionMatrix(test_data,pred_data))
    print("precision Result",generatePrecisionResult(test_data,pred_data))
    print("Recall Result",generateRecallResult(test_data,pred_data))
    print("F1 Measure ", generateF1MeasureResult(test_data,pred_data))
    

