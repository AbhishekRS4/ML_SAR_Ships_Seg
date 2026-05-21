import logging
import argparse

from utils.tfrecords_generator import create_tfrecords


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--dir-images",
        default=None,
        type=str,
        help="full path to the directory containing images",
    )
    arg_parser.add_argument(
        "--dir-labels",
        default=None,
        type=str,
        help="full path to the directory containing labels",
    )
    arg_parser.add_argument(
        "--dir-output",
        default=None,
        type=str,
        help="full path to the directory where output tfrecord files need to be created",
    )
    arg_parser.add_argument(
        "--num-shards",
        default=8,
        type=int,
        help="num shards into which the tfrecord files need to be created",
    )
    arg_parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="the number of workers that needs to be used for process pool executor",
    )
    arg_parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to validate image and label files or not",
    )
    arg_parser.add_argument(
        "--out-log-file",
        default="tfrecords_generator.log",
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

    create_tfrecords(
        ARGS.dir_images,
        ARGS.dir_labels,
        ARGS.dir_output,
        num_shards=ARGS.num_shards,
        num_workers=ARGS.num_workers,
        validate=ARGS.validate,
    )

    return


if __name__ == "__main__":
    main()
