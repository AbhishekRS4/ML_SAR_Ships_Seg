import logging
import argparse


from utils.label_processing import hrsid_json_to_lbl_image


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--file-json",
        default=None,
        type=str,
        help="full path to the json file with annotations",
    )
    arg_parser.add_argument(
        "--dir-output-lbls",
        default=None,
        type=str,
        help="full path to the directory where labels need to be saved",
    )
    arg_parser.add_argument(
        "--label-type",
        default="semantic_seg",
        choices=["semantic_seg", "instance_seg"],
        type=str,
        help="the type of label that needs to be saved",
    )
    arg_parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="the number of workers that needs to be used for thread pool executor",
    )
    arg_parser.add_argument(
        "--out-log-file",
        default="lbl_image_creator.log",
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

    hrsid_json_to_lbl_image(
        ARGS.file_json,
        ARGS.dir_output_lbls,
        label_type=ARGS.label_type,
        num_workers=ARGS.num_workers,
    )
    return


if __name__ == "__main__":
    main()
