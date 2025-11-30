import logging
import argparse


from utils.move_data_for_training import move_ship_images


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--dir-ship-images",
        default=None,
        type=str,
        help="full path to the directory where pure background images are present",
    )
    arg_parser.add_argument(
        "--dir-train-images",
        default=None,
        type=str,
        help="full path to the directory where train images are to be moved",
    )
    arg_parser.add_argument(
        "--dir-train-labels",
        default=None,
        type=str,
        help="full path to the directory where train labels are to be moved",
    )
    arg_parser.add_argument(
        "--dir-test-images",
        default=None,
        type=str,
        help="full path to the directory where test images are to be moved",
    )
    arg_parser.add_argument(
        "--dir-test-labels",
        default=None,
        type=str,
        help="full path to the directory where test labels are to be moved",
    )
    arg_parser.add_argument(
        "--out-log-file",
        default="move_data_for_training.log",
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

    move_ship_images(
        ARGS.dir_ship_images,
        ARGS.dir_train_images,
        ARGS.dir_train_labels,
        ARGS.dir_test_images,
        ARGS.dir_test_labels,
    )
    return


if __name__ == "__main__":
    main()
