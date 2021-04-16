import torch
import os
import cv2
from path import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from torch import nn, optim
from Code import constants
from Code import metricsEvaluation
import gc

def ImageDataSetPrep(data_type):
    if data_type == "test":
        path_test_train = constants.TEST
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path+ "/"+ constants.Masked_Folder_test
        unmasked_Path = base_Path+"/"+ constants.UnMasked_Folder_test
        nonPerson_Path = base_Path+"/"+ constants.NonPerson_Folder_test
    else:
        path_test_train = constants.Train
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path +"/"+ constants.Masked_Folder_train
        unmasked_Path = base_Path+"/"+ constants.UnMasked_Folder_train
        nonPerson_Path = base_Path+"/"+ constants.NonPerson_Folder_train

    if len(os.listdir(masked_Path)) != 0 or len(os.listdir(unmasked_Path)) != 0 or len(os.listdir(nonPerson_Path)) != 0:
        path_dirs = [[nonPerson_Path, 2], [masked_Path, 1], [unmasked_Path, 0]]
        if not os.path.exists(base_Path):
            raise Exception("The data path doesn't exist")
        for Dir_path, label in path_dirs:
            for image in os.listdir(Dir_path):
                imagePath=os.path.normpath(Dir_path)
                imagePath = os.path.join(imagePath, image)
                try:
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (constants.imageSize, constants.imageSize))
                    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    tempImg=img
                    tempImage=os.path.splitext(image)[0]
                    image=tempImage
                    if label == 2:
                        cv2.imwrite(path_test_train + "\class2\\" + image + ".png", tempImg)
                    if label == 1:
                        cv2.imwrite(path_test_train + "\class1\\" + image + ".png", tempImg)
                    if label == 0:
                        cv2.imwrite(path_test_train + "\class0\\" + image + ".png", tempImg)
                except:
                    print('Error Occured while processing images',imagePath)
                    pass


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=100,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1),
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
    ImageDataSetPrep("test")
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(root=constants.TEST, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
    PATH = "trained_cnn_Model.pt"
    if not os.path.exists(PATH):
        ImageDataSetPrep("train")
        data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])
        trainset = torchvision.datasets.ImageFolder(root=constants.Train, transform=data_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

        for epoch in range(20):
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                outputs = net.forward(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #TEST ACCURACY
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #END
                gc.collect()
                torch.cuda.empty_cache()
            print('Accuracy of the network on Epoch',epoch, ': %d %%' % (
            100 * correct / total))

        # Save
        torch.save(net, PATH)
    else:
        net = torch.load(PATH)
    pred_data = []
    test_data = []
    correct = 0
    total = 0
    
    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = net(images)
        test_data.extend(labels.numpy().tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        pred_data.extend(predicted.numpy().tolist())
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on ',len(test_data), ' test images: %d %%' % (
            100 * correct / total))
    print("\n")
    
    metricsEvaluation.evaluateCNNModel(test_data, pred_data)
    
