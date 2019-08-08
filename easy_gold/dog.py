import chainer
from chainer.dataset import dataset_mixin
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
import chainercv
from collections import defaultdict
import glob
import os
import numpy as np
import xml.etree.ElementTree as ET


class DogDataset(dataset_mixin.DatasetMixin):
    def __init__(self, crop=False, size=32, **kwargs):
        root = '../input/all-dogs/all-dogs/'
        paths = sorted(os.listdir(root))
        self.crop = crop
        self.size = size
        if self.crop:
            self._dataset = DogCropDataset()
        else:
            self._dataset = chainer.datasets.ImageDataset(paths, root=root)

    def __len__(self):
        return len(self._dataset)
    
    def get_example(self, i):
        if self.crop:
            img, bbox, label = self._dataset[i]
            # TODO: translation
            ymin, xmin, ymax, xmax = bbox
            img = img[:, ymin:ymax, xmin:xmax]
        else:
            img = self._dataset[i]
            label = 0

        # img = chainercv.transforms.resize(img, (32, 32))
        img = chainercv.transforms.scale(img, self.size, fit_short=True)
        img = chainercv.transforms.random_crop(img, (self.size, self.size))
        img = chainercv.transforms.random_flip(img, x_random=True)
        img = (img / 128. - 1.).astype(np.float32)
        img += np.random.uniform(size=img.shape, low=0., high=1. / 128)
        return img, label


class DogBBoxDataset(GetterDataset):
    def __init__(self):
        super(DogBBoxDataset, self).__init__()
        root_image = '../input/all-dogs/all-dogs/'
        root_annot = '../input/annotation/Annotation/'
        annots = glob.glob(root_annot + '*/*')
        annots = sorted(annots)
        breeds = os.listdir(root_annot)
        breeds = ['-'.join(breed.split('-')[1:]) for breed in breeds]
        self.names = list(set(breeds))
        self.image_annot_dict = defaultdict(list)
        for annot in annots:
            annot_ = annot.split('/')
            breed, path = annot_[:-1], annot_[-1]
            self.image_annot_dict[path + '.jpg'].append(annot)
        
        image_paths = sorted(list(self.image_annot_dict.keys()))
        # no image for ../input/all-dogs/all-dogs/n02105855_2933.jpg
        image_paths = [path for path in image_paths if os.path.isfile(os.path.join(root_image, path))]
        self._dataset = chainer.datasets.ImageDataset(image_paths, root=root_image)

        self.add_getter('image', self.get_image)
        self.add_getter(('bbox', 'label'), self.get_annotation)

    def __len__(self):
        return len(self._dataset)
    
    def get_image(self, i):
        img = self._dataset[i]
        return img
    
    def get_annotation(self, i):
        path = self._dataset._paths[i]
        annots = self.image_annot_dict[path]

        bbox = list()
        label = list()
        for annot in annots:
            tree = ET.parse(annot)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox')
                ymin = int(bndbox.find('ymin').text)
                xmin = int(bndbox.find('xmin').text)
                ymax = int(bndbox.find('ymax').text)
                xmax = int(bndbox.find('xmax').text)
                bbox.append((ymin, xmin, ymax, xmax))
                nm = o.find('name')
                label.append(self.names.index(nm.text))
        
        bbox = np.array(bbox)
        label = np.array(label)
        return bbox, label


class DogCropDataset(dataset_mixin.DatasetMixin):
    def __init__(self):
        self.dataset = DogBBoxDataset()
        self.names = self.dataset.names
        self.indices = list()
        self.bboxes = list()
        self.labels = list()
        for i in range(len(self.dataset)):
            bbox, label = self.dataset.get_example_by_keys(i, (1, 2))
            self.indices.append(np.ones_like(label) * i)
            self.bboxes.append(bbox)
            self.labels.append(label)
        
        self.indices = np.concatenate(self.indices, axis=0)
        self.bboxes = np.concatenate(self.bboxes, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def get_example(self, i):
        idx = self.indices[i]
        img, = self.dataset.get_example_by_keys(idx, (0,))
        bbox, label = self.bboxes[i], self.labels[i]
        return img, bbox, label
