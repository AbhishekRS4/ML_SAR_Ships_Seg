import logging
import argparse


from utils.pure_bg_label_processing import create_pure_bg_lbl_images


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--dir-pure-bg-images",
        default=None,
        type=str,
        help="full path to the directory where pure background images are present",
    )
    arg_parser.add_argument(
        "--dir-pure-bg-out-labels",
        default=None,
        type=str,
        help="full path to the directory where labels need to be saved",
    )
    arg_parser.add_argument(
        "--image-height",
        default=800,
        type=int,
        help="the image height of the pure background images",
    )
    arg_parser.add_argument(
        "--image-width",
        default=800,
        type=int,
        help="the image width of the pure background images",
    )
    arg_parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="the number of workers that needs to be used for thread pool executor",
    )
    arg_parser.add_argument(
        "--out-log-file",
        default="pure_bg_lbl_image_creator.log",
        type=str,
        help="the log file where all the logs are recorded",
    )

    ARGS, _ = arg_parser.parse_known_args()
    return ARGS


def main() -> None:
    ARGS = parse_args()

    logging.basicConfig(
        filename=ARGS.out_log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    create_pure_bg_lbl_images(
        ARGS.dir_pure_bg_images,
        ARGS.dir_pure_bg_out_labels,
        image_height=ARGS.image_height,
        image_width=ARGS.image_width,
        num_workers=ARGS.num_workers,
    )
    return


if __name__ == "__main__":
    main()
