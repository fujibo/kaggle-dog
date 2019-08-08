import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
import chainercv
from PIL import Image
import shutil
from chainercv.links.model.ssd.transforms import resize_with_random_interpolation
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images, gen_images_with_condition
import yaml
import yaml_utils


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='/kaggle')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--post_proc', choices={'bilinear', 'bicubic', 'random', None}, default=None)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu(args.gpu)
    out = args.results_dir

    if not os.path.exists(os.path.join(out, 'images')):
        os.makedirs(os.path.join(out, 'images'))

    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    n_samples_per_class = [84] * 40 + [83] * 80
    assert sum(n_samples_per_class) == args.n_samples
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        if args.conditional:
            x = list()
            for c, n_samples in enumerate(n_samples_per_class):
                x_c = gen_images_with_condition(gen, c=c, n=n_samples, batchsize=n_samples)
                x.append(x_c)
            x = np.concatenate(x, axis=0)
            
        else:
            x = gen_images(gen, n=args.n_samples, batchsize=100)
    
    n, c, h, w = x.shape
    for i, img in enumerate(x):
        if args.post_proc in {'bilinear', 'bicubic'}:
            if args.post_proc == 'bilinear':
                interpolation = Image.BILINEAR
            else:
                interpolation = Image.BICUBIC
            img_res = chainercv.transforms.resize(img, (64, 64), interpolation=interpolation)
            img_res = np.clip(img_res, 0.0, 255.0)
        elif args.post_proc == 'random':
            img_res = resize_with_random_interpolation(img, (64, 64))
        elif args.post_proc is None:
            img_res = img
        else:
            raise NotImplementedError
        chainercv.utils.write_image(
            img_res,
            os.path.join(out, 'images', 'image_{:05d}.png'.format(i)))

    shutil.make_archive('images', 'zip', os.path.join(out, 'images'))

    # rows, columns = 100, args.n_samples // 100
    # x = x.reshape((rows, columns, 3, h, w))
    # x = x.transpose(0, 3, 1, 4, 2)
    # x_all = x.reshape((rows * h, columns * w, 3))
    # chainercv.utils.write_image(x_all.transpose(2, 0, 1), '/kaggle/working/all.png')


if __name__ == '__main__':
    main()
