import logging
import argparse

from inference.infer import aot_inductor_inference_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir-infer-images",
        default="HRSID/test_images",
        type=str,
        help="full path to the directory with the inference set images",
    )
    parser.add_argument(
        "--dir-pred-labels",
        default=None,
        type=str,
        help="full path to the directory where the predicted labels need to be saved",
    )
    parser.add_argument(
        "--file-model-ckpt",
        default=None,
        type=str,
        help="full path to the checkpoint file to load for finetuning the model",
    )
    parser.add_argument(
        "--which-gpu",
        default="0",
        type=str,
        help="which GPU needs to be used for training",
    )
    parser.add_argument(
        "--out-log-file",
        default="hrsid_sem_seg_inference.log",
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

    aot_inductor_inference_pipeline(
        ARGS.dir_infer_images,
        ARGS.dir_pred_labels,
        ARGS.file_model_ckpt,
        which_gpu=ARGS.which_gpu,
    )
    return


if __name__ == "__main__":
    main()
