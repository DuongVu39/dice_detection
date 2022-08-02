from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def train(self, do_augment: bool = False):
        if do_augument:
            self.augmentation()
        pass

    @abstractmethod
    def augmentation(self):
        pass

    @abstractmethod
    def validation(self):
        pass


class VggCnnNumberClassifierTrainer(Trainer):

    def __init__(self, X_train, y_train):
        super(VggCnnNumberClassifierTrainer, self).__init__(X_train, y_train)

    def train(self, do_augment: bool = False):
        pass

