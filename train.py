from unet import Unet
from utils import read_data
import tensorflow as tf
import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=640)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--train_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/train')
    parser.add_argument('--mask_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/train_masks')
    parser.add_argument('--n_images', type=int, default=0)

    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)

    return parser


def main():
    args = build_parser().parse_args()
    image_size = [args.img_height, args.img_width]
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # sess = tf.Session(config=config)
    sess = tf.Session()
    unet = Unet(input_shape=image_size, sess=sess, filter_num=args.filter_num)
    unet.build_net()
    if args.checkpoint_path:
        unet.load_weights(args.checkpoint_path)

    images, masks = read_data(args.train_dir,
                              args.mask_dir,
                              n_images=args.n_images, image_size=image_size)
    unet.train(images, masks, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
