from unet import Unet
from utils import read_data
import tensorflow as tf
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=640)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--batch_norm', action='store_true', default=False)

    # checkpoint path
    parser.add_argument('--checkpoint_path', type=str, default=None)

    # train data
    parser.add_argument('--train_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/train_val/train')
    parser.add_argument('--train_mask_dir', type=str,
                        default='/home/wangxiyang/dataset/kaggle/data/train_val/train_mask')
    parser.add_argument('--val_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/train_val/val')
    parser.add_argument('--val_mask_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/train_val/val_mask')
    parser.add_argument('--n_images', type=int, default=0)

    # train parameters
    parser.add_argument('--dice_loss', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--always_save', action='store_true', default=False)

    return parser


def main():
    args = build_parser().parse_args()
    image_size = [args.img_height, args.img_width]
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # sess = tf.Session(config=config)
    sess = tf.Session()
    unet = Unet(input_shape=image_size, sess=sess, filter_num=args.filter_num, batch_norm=args.batch_norm)
    unet.build_net()
    if args.checkpoint_path:
        unet.load_weights(args.checkpoint_path)

    images, masks = read_data(args.train_dir,
                              args.train_mask_dir,
                              n_images=args.n_images, image_size=image_size)
    val_images, val_masks = read_data(args.val_dir,
                                      args.val_mask_dir,
                                      n_images=args.n_images // 4, image_size=image_size)
    unet.train(images=images, masks=masks, val_images=val_images, val_masks=val_masks, epochs=args.epochs,
               batch_size=args.batch_size, learning_rate=args.learning_rate, dice_loss=args.dice_loss,
               always_save=args.always_save)


if __name__ == '__main__':
    main()
