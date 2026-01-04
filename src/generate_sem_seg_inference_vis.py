import logging
import argparse


from utils.visualization import save_visualization_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir-images",
        default="HRSID/test_images",
        type=str,
        help="full path to the directory containing the images",
    )
    parser.add_argument(
        "--dir-gt-lbls",
        default="HRSID/test_labels",
        type=str,
        help="full path to the directory containing the GT labels",
    )
    parser.add_argument(
        "--dir-pred-lbls",
        default=None,
        type=str,
        help="full path to the directory containing the predicted labels",
    )
    parser.add_argument(
        "--dir-vis",
        default=None,
        type=str,
        help="full path to the directory where the visualizations need to be saved",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="num workers for the ProcessPoolExecutor for multiprocessing",
    )
    parser.add_argument(
        "--out-log-file",
        default="gen_sem_seg_inference_vis.log",
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

    save_visualization_pipeline(
        ARGS.dir_images,
        ARGS.dir_gt_lbls,
        ARGS.dir_pred_lbls,
        ARGS.dir_vis,
        num_workers=ARGS.num_workers,
    )
    return


if __name__ == "__main__":
    main()
