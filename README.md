# SAR Ships Segmentation

## Info
* This repo contains ship segmentation in the processed images extracted from the Synthetic Aperture Radar (SAR) data. The dataset used in this repo is [HRSID](https://github.com/chaozhong2010/HRSID)
* This project involves learning some advanced concepts like various model training and inference optimization strategies, using TFRecords for dataset creation, training using Nvidia Dali Pipelines, DDP training using torchrun


## Instructions to run code for data preparation
* Run [src/create_label_images.py](src/create_label_images.py) to generate label images from the COCO style format JSON labels
* Run [src/create_pure_bg_label_images.py](src/create_pure_bg_label_images.py) to generate pure background label images
* Run [src/move_ship_images_for_training.py](src/move_ship_images_for_training.py) to move the ship images to train and test set directories to be used for training
* Run [src/move_pure_bg_data_for_training.py](src/move_pure_bg_data_for_training.py) to move the pure background images and labels to train and test set directories to be used for training
* Run [src/create_tfrecords.py](src/create_tfrecords.py) to create TFRecords dataset files


## Instructions to optimize model checkpoints for inference
* Run [src/optimize_model_with_aot_inductor.py](src/optimize_model_with_aot_inductor.py) to optimize the model checkpoint with AOT Inductor compilation. This is mainly useful to reduce the inference cold start latence i.e. to reduce the time for the first inference
* Run [src/optimize_model_with_onnx.py](src/optimize_model_with_onnx.py) to optimize the model checkpoint with ONNX
* Run [src/optimize_model_with_tensorrt.py](src/optimize_model_with_tensorrt.py) to optimize the model checkpoint with TensorRT


## Instructions to run code for ship segmentation
* Run [src/run_sem_seg_trainer.py](src/run_sem_seg_trainer.py) to run training experiments for PNG
* Run [src/run_sem_seg_trainer_dali_png.py](src/run_sem_seg_trainer_dali_png.py) to run training experiments with Dali pipeline for PNG
* Run [src/run_sem_seg_trainer_dali_tfrecords.py](src/run_sem_seg_trainer_dali_tfrecords.py) to run training experiments with Dali pipeline for TFRecords
* Run [src/run_sem_seg_trainer_ddp_dali_tfrecords.py](src/run_sem_seg_trainer_ddp_dali_tfrecords.py) to run DDP multi-gpu training experiments with Dali pipeline for TFRecords
* Run [src/run_sem_seg_inference_torch.py](src/run_sem_seg_inference_torch.py) to run inference
* Run [src/run_sem_seg_eval.py](src/run_sem_seg_eval.py) to run evaluation
* Run [src/generate_sem_seg_inference_vis.py](src/generate_sem_seg_inference_vis.py) to generate visualizations of the predictions


## Label preparation code optimization
* The label images are prepared using the COCO format JSON label files
* The optimization used is the `ThreadPoolExecutor` to generate label images using multiple worker threads. The choice of `ThreadPoolExecutor` is due to the nature of the task that involes I/O bound operations i.e. saving the images to the disk and the operation is thread-safe
* The following table shows the comparison of speedup with optimized code

| Optimization method | Avg. Time taken (in sec.) |
| ------------------- | ------------------------- |
| Without any optimization |  21.80  |
| With ThreadPoolExecutor  |   9.32  |


## Model training optimization using torch.compile
* The torch.compile can be used for optimizing the model training that can effectively result in reduction in the training time
* The following table shows the training time per epoch for the **ConvNextV2-Tiny-DeepLabV3+** model

| Model training + compile method | Batch Size (per GPU) | Data loading method | Avg. time per epoch (in sec.) |
| --------------------------------| -----------| ------------------- | ----------------------------- |
| Single GPU with Reduce-overhead |  12        | Torch DataLoader with PNG image loading  |   520   |
| Single GPU with Max-Autotune    |   8        | Torch DataLoader with PNG image loading  |   480   |
| Single GPU with Reduce-overhead |  8        | Dali Pipeline with TFRecords             |    440   |
| Single GPU with Reduce-overhead |  8        | Dali Pipeline with PNG                   |    485   |
| DDP + Multi GPUs (2) with Reduce-overhead |  8  | Dali Pipeline with TFRecords |   220   |


## DDP model training on a single node with multiple GPUs
* The following method is recommended for DDP training even on single node with multiple GPUs (denoted by `nproc_per_node` param in the `torchrun` cmd)
```
torchrun --standalone --nproc_per_node=2 ML_SAR_Ships/src/run_sem_seg_trainer_ddp_dali_tfrecords.py --num-epochs 10 --learning-rate 3e-5 ...
```


## Model inference using AOT inductor
* The AOT inductor model compilation can result in the reduction in the initial inference start time i.e. the inference cold start can be reduced significantly. The inference time without preprocessing and data transfer
* The following table shows the starting inference time speedup with AOT inductor model compilation for the **ConvNextV2-Tiny-DeepLabV3+** model. It can be clearly observed that AOT inductor compiled model reduced the starting inference time by a significant amount

| Model method | Starting [first, second data samples] inference time (milli sec.) | Avg. inference time [other data samples] (milli sec.) |
| ------------ | ------------------------------------ | ---------------|
| Without AOT inductor compilation | 3645.17, 210.14 | 30.5 |
| With AOT inductor compilation | 186.1, 30.56 | 30.5 |


## Model inference using ONNX
* The ONNX model optimization is used to benchmark the inference time on the same GPU
* The following table shows the starting inference time speedup with ONNX model optimization for the **ConvNextV2-Tiny-DeepLabV3+** model. It can be clearly seen that the avg inference time with ONNX model is higher than that of normal torch inference and AOT inductor compiled inference

| Model method | Starting [first, second data samples] inference time (milli sec.) | Avg. inference time [other data samples] (milli sec.) |
| ------------ | ------------------------------------ | ---------------|
| ONNX model | 631.04, 136.03 | 102.3 |
| ONNX model with IO binding | 634.98, 135.54 | 102.00 |


## Model inference using TensorRT
* The TensorRT model optimization is used to benchmark the inference time on the same GPU. The inference time without preprocessing and data transfer
* For performing any experiments with TensorRT, use the following docker image - `nvcr.io/nvidia/pytorch:25.11-py3`
* Refer [docker_cmds/README.md](docker_cmds/README.md) for instructions to run TensorRT experiments
* The dynamo frontend is used to compile the export program in both fp32 and mixed precisions for the **ConvNextV2-Tiny-DeepLabV3+** model. However, the mixed precision did not work since the inference results were all blank with mixed precision. Only fp32 worked with TensorRT
* The following table shows the inference time with TensorRT optimization. It can be clearly observed that it did not give a significant improvement in the inference time

| Model precision | Avg. inference time (milli sec.) |
| ------------ | ------------------------------------ |
| float32 | 37.5 |


## Visualization generation optimization
* The visualization generation pipeline is optimized using `ProcessPoolExecutor`. This is the preferred choice since Matplotlib is not thread-safe
* The following table shows the time taken for generating the visualizations for the test set containing around 2000 images

| Num workers | Time taken (in sec.) |
| ----------- | -------------------- |
| 4 | 169.81 |
| 8 | 86.65 |


## Model performance quantitative metrics
* The following table shows the quantitative metrics of the ConvNextV2-Tiny-DeepLabV3+ model performance without using class weights in the Focal loss function. The models have not been trained well enough to optimize for performance metrics but for learning new skills.

| Model name | Train mIoU | Train Dice | Test mIoU | Test Dice |
| ---------- | ---------- | ---------- | --------- | --------- |
| ConvNextV2-Tiny-DeepLabV3+ | 0.8116 | 0.9978 | 0.8086 | 0.9970 |
| ResNet34-UNet              | 0.8959 | 0.9990 | 0.8867 | 0.9983 |
| PSAResNet34-UNet           | 0.8776 | 0.9986 | 0.8725 | 0.9977 |


## Model performance qualitative visualization results - sample test set predictions
* The following seven visualizations show the qualitative results of the ConvNextV2-Tiny-DeepLabV3+ model performance
![Sample predicted mask 1](images/ConvNextV2-Tiny-DeepLabV3+/P0001_2400_3200_4800_5600.png?raw=true)
![Sample predicted mask 2](images/ConvNextV2-Tiny-DeepLabV3+/P0011_600_1400_8189_8989.png?raw=true)
![Sample predicted mask 3](images/ConvNextV2-Tiny-DeepLabV3+/P0017_600_1400_8400_9200.png?raw=true)
![Sample predicted mask 4](images/ConvNextV2-Tiny-DeepLabV3+/P0025_1200_2000_10189_10989.png?raw=true)
![Sample predicted mask 5](images/ConvNextV2-Tiny-DeepLabV3+/P0063_1200_2000_7800_8600.png?raw=true)
![Sample predicted mask 6](images/ConvNextV2-Tiny-DeepLabV3+/P0082_3000_3800_600_1400.png?raw=true)
![Sample predicted mask 7](images/ConvNextV2-Tiny-DeepLabV3+/P0083_1800_2600_1800_2600.png?raw=true)
* The following visualization show the results of different inference methods to show that the inference predictions are consistent across different inference methods using ConvNextV2-Tiny-DeepLabV3+ model
![Sample predicted mask 8](images/ConvNextV2-Tiny-DeepLabV3+/inference_results_comparison.png?raw=true)
* The following seven visualizations show the qualitative results of the ResNet34-UNet model performance
![Sample predicted mask 1](images/ResNet34-UNet/P0001_2400_3200_4800_5600.png?raw=true)
![Sample predicted mask 2](images/ResNet34-UNet/P0011_600_1400_8189_8989.png?raw=true)
![Sample predicted mask 3](images/ResNet34-UNet/P0017_600_1400_8400_9200.png?raw=true)
![Sample predicted mask 4](images/ResNet34-UNet/P0025_1200_2000_10189_10989.png?raw=true)
![Sample predicted mask 5](images/ResNet34-UNet/P0063_1200_2000_7800_8600.png?raw=true)
![Sample predicted mask 6](images/ResNet34-UNet/P0082_3000_3800_600_1400.png?raw=true)
![Sample predicted mask 7](images/ResNet34-UNet/P0083_1800_2600_1800_2600.png?raw=true)