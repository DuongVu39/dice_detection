import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
import torchvision


class VggCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3)
        self.max_pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3)
        # TODO: FIGURE OUT DIMENSIONS AT THIS POINT
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(self.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(self.relu(self.conv4(F.relu(self.conv3(x)))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

    # np.random.seed(1337)  # for reproducibility
    #
    # # input image dimensions
    # img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    #
    # # size of pooling area for max pooling
    # pool_size = (2, 2)
    # # convolution kernel size
    # kernel_size = (3, 3)
    # input_shape = (img_rows, img_cols, 1)
    #
    # model = Sequential()
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    #                         border_mode='valid',
    #                         input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # # (16, 8, 32)
    #
    # model.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # # (8, 4, 64) = (2048)
    #
    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])
