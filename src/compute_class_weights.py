import logging
import argparse

from utils.class_imbalance import compute_class_weights


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir-labels",
        default="/home/abhishek/datasets/HRSID/train_labels/",
        type=str,
        help="full path to the directory with the label image files",
    )
    parser.add_argument(
        "--out-log-file",
        default="class_imbalance.log",
        type=str,
        help="the log file where all the logs are recorded",
    )

    ARGS, unparsed = parser.parse_known_args()
    return ARGS


def main() -> None:
    ARGS = parse_arguments()

    logging.basicConfig(
        filename=ARGS.out_log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    compute_class_weights(
        ARGS.dir_labels,
    )
    return


if __name__ == "__main__":
    main()
