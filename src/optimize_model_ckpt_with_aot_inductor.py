import logging
import argparse

from inference.model_optimization import optimize_model_with_aot_inductor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file-model-ckpt",
        default=None,
        type=str,
        help="full path to the checkpoint file to load for finetuning the model",
    )
    parser.add_argument(
        "--model-name",
        default="convnext_v2_tiny_deeplab_v3+",
        choices=["convnext_v2_tiny_deeplab_v3+", "convnext_v2_base_deeplab_v3+"],
        type=str,
        help="the model that needs to be trained",
    )
    parser.add_argument(
        "--image-height",
        default=800,
        type=int,
        help="image height for AOT inductor sample input",
    )
    parser.add_argument(
        "--image-width",
        default=800,
        type=int,
        help="image width for AOT inductor sample input",
    )
    parser.add_argument(
        "--model-compile-mode",
        default="reduce-overhead",
        type=str,
        choices=["normal", "reduce-overhead", "max-autotune", "uncompiled"],
        help="indicates the model compile option to use to reduce overhead during training",
    )
    parser.add_argument(
        "--which-gpu",
        default="0",
        type=str,
        help="which GPU needs to be used for training",
    )
    parser.add_argument(
        "--out-log-file",
        default="optimize_model_with_aot_inductor.log",
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

    optimize_model_with_aot_inductor(
        ARGS.file_model_ckpt,
        ARGS.model_name,
        ARGS.image_height,
        ARGS.image_width,
        model_compile_mode=ARGS.model_compile_mode,
        which_gpu=ARGS.which_gpu,
    )
    return


if __name__ == "__main__":
    main()
