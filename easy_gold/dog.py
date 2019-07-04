import chainer
from chainer.dataset import dataset_mixin
import chainercv
import os


class DogDataset(dataset_mixin.DatasetMixin):
    def __init__(self, **kwargs):
        root = '../input/all-dogs/all-dogs/'
        paths = sorted(os.listdir(root))
        self._dataset = chainer.datasets.ImageDataset(paths, root=root)

    def __len__(self):
        return len(self._dataset)
    
    def get_example(self, i):
        img = self._dataset[i]
        img = chainercv.transforms.resize(img, (32, 32))
        img = chainercv.transforms.random_flip(img, x_random=True)
        return img
            