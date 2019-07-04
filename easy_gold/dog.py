import chainer
from chainer.dataset import dataset_mixin
import os


class DogDataset(dataset_mixin.DatasetMixin):
    def __init__(self):
        root = '../input/all-dogs/all-dogs/'
        paths = sorted(os.listdir(root))
        self._dataset = chainer.datasets.ImageDataset(paths, root=root)

    def __len__(self):
        return len(self._dataset)
    
    def get_example(self, i):
        # TODO: flip
        return self._dataset[i]