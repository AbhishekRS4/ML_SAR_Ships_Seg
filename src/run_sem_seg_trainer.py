import logging
import argparse

from trainer.train_sem_seg import train_sem_seg_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir-train-images",
        default="HRSID/train_images",
        type=str,
        help="full path to the directory with the train set images",
    )
    parser.add_argument(
        "--dir-train-labels",
        default="HRSID/train_labels",
        type=str,
        help="full path to the directory with the train set labels",
    )
    parser.add_argument(
        "--dir-test-images",
        default="HRSID/test_images",
        type=str,
        help="full path to the directory with the test set images",
    )
    parser.add_argument(
        "--dir-test-labels",
        default="HRSID/test_labels",
        type=str,
        help="full path to the directory with the test set labels",
    )
    parser.add_argument(
        "--dir-tmp-ckpt-model",
        default="tmp_ckpt_model",
        type=str,
        help="full path to the directory where model checkpoint files need to be stored temporarily that will be logged to MLflow tracking server",
    )
    parser.add_argument(
        "--batch-size",
        default=12,
        type=int,
        help="batch size to be used for training",
    )
    parser.add_argument(
        "--num-classes",
        default=2,
        type=int,
        help="number of target classes in the dataset",
    )
    parser.add_argument(
        "--labels-display-logs",
        nargs="*",
        default=[
            "background",
            "ship",
        ],
        help="the list of labels for every class label that needs to be used for logging",
    )
    parser.add_argument(
        "--class-weights",
        nargs="*",
        default=[
            0.50160638,
            156.12963969,
        ],
        type=float,
        help="class weights to be applied in the loss function during training",
    )
    parser.add_argument(
        "--experiment-name",
        default="HRSID_SAR",
        type=str,
        help="the experiment name in MLFlow",
    )
    parser.add_argument(
        "--run-name",
        default="HRSID_SAR",
        type=str,
        help="the run name in MLFlow",
    )
    parser.add_argument(
        "--model-name",
        default="convnext_v2_tiny_deeplab_v3+",
        choices=[
            "convnext_v2_tiny_deeplab_v3+",
            "convnext_v2_base_deeplab_v3+",
            "resnet34_unet",
        ],
        type=str,
        help="the model that needs to be trained",
    )
    parser.add_argument(
        "--loss-fn",
        default="focal",
        choices=[
            "cross_entropy",
            "focal",
        ],
        type=str,
        help="the loss function to be used for training the model",
    )
    parser.add_argument(
        "--optimizer-name",
        default="adamw",
        choices=["adamw", "sgd"],
        type=str,
        help="the optimizer to be used for training the model",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="initial learning rate to be used for training",
    )
    parser.add_argument(
        "--weight-decay",
        default=5e-5,
        type=float,
        help="weight decay",
    )
    parser.add_argument(
        "--num-epochs",
        default=200,
        type=int,
        help="number of epochs for which the model needs to be trained",
    )
    parser.add_argument(
        "--checkpoint-freq",
        default=2,
        type=int,
        help="checkpoint epoch frequency for which the model needs to be logged",
    )
    parser.add_argument(
        "--checkpoint-skip",
        default=10,
        type=int,
        help="checkpoint epoch skip is the first few epochs for which the model need not be logged",
    )
    parser.add_argument(
        "--which-gpu",
        default="0",
        type=str,
        help="which GPU needs to be used for training",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="num workers to be used in the dataloaders (choose num_workers <= batch_size, physical CPU cores)",
    )
    parser.add_argument(
        "--num-in-channels",
        default=1,
        type=int,
        help="num input channels to the model",
    )
    parser.add_argument(
        "--model-compile-mode",
        default="max-autotune",
        type=str,
        choices=["reduce-overhead", "max-autotune", "uncompiled"],
        help="indicates the model compile option to use to reduce overhead during training",
    )
    parser.add_argument(
        "--file-model-ckpt",
        default=None,
        type=str,
        help="full path to the checkpoint file to load for finetuning the model",
    )
    parser.add_argument(
        "--out-log-file",
        default="hrsid_sem_seg_trainer.log",
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

    train_sem_seg_pipeline(
        ARGS.dir_train_images,
        ARGS.dir_train_labels,
        ARGS.dir_test_images,
        ARGS.dir_test_labels,
        ARGS.dir_tmp_ckpt_model,
        ARGS.batch_size,
        ARGS.num_classes,
        ARGS.labels_display_logs,
        ARGS.class_weights,
        ARGS.experiment_name,
        ARGS.run_name,
        ARGS.model_name,
        ARGS.loss_fn,
        ARGS.optimizer_name,
        learning_rate=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        num_epochs=ARGS.num_epochs,
        checkpoint_freq=ARGS.checkpoint_freq,
        checkpoint_skip=ARGS.checkpoint_skip,
        which_gpu=ARGS.which_gpu,
        num_workers=ARGS.num_workers,
        num_in_channels=ARGS.num_in_channels,
        model_compile_mode=ARGS.model_compile_mode,
        file_model_ckpt=ARGS.file_model_ckpt,
    )
    return


if __name__ == "__main__":
    main()
