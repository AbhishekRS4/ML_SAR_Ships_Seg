# SARShips

## Info
* The project is regarding the segmentation of ships in the processed images extracted from the Synthetic Aperture Radar (SAR) data
* This project involves learning some new things like using KServe for deployment of the ML model along with other model optimization strategies

## Instructions to run code for data preparation
* Run [src/create_label_images.py](src/create_label_images.py) to generate label images from the COCO style format JSON labels
* Run [src/create_pure_bg_label_images.py](src/create_pure_bg_label_images.py) to generate pure background label images
* Run [src/move_ship_images_for_training.py](src/move_ship_images_for_training.py) to move the ship images to train and test set directories to be used for training
* Run [src/move_pure_bg_data_for_training.py](src/move_pure_bg_data_for_training.py) to move the pure background images and labels to train and test set directories to be used for training

## Instructions to optimize model checkpoints for inference
* Run [src/optimize_model_ckpt_with_aot_inductor.py](src/optimize_model_ckpt_with_aot_inductor.py) to optimize the model checkpoint with AOT Inductor compilation. This is mainly useful to reduce the inference cold start latence i.e. to reduce the time for the first inference

## Instructions to run code for semantic segmentation task
* Run [src/run_sem_seg_trainer.py](src/run_sem_seg_trainer.py) to run training experiments for the semantic segmentation task
* Run [src/run_sem_seg_inference.py](src/run_sem_seg_inference.py) to run inference for the semantic segmentation task
* Run [src/generate_sem_seg_inference_vis.py](src/generate_sem_seg_inference_vis.py) to generate visualizations of the predictions

## Label preparation code optimization
* The label images are prepared using the COCO format JSON label files
* The optimization used is the ThreadPoolExecutor to generate label images using multiple worker threads. The choice of ThreadPoolExecutor is due to the nature of the task that involes I/O bound operations i.e. saving the images to the disk
* The following table shows the comparison of speedup with optimized code

| Optimization Method | Time taken (in sec.) |
| ------------------- | -------------------- |
| Without any optimization |  21.80  |
| With ThreadPoolExecutor  |   9.32  |

## Model optimization using AOT inductor
* The AOT inductor model compilation can result in the reduction in the initial inference start time
* The following table shows the starting inference time speedup with AOT inductor model compilation

| Model method | Starting inference time (milli sec.) |
| ------------ | ------------------------------------ |
| Without AOT inductor compilation | 3621 |
| With AOT inductor compilation | 179.69 |

## Model performance quantitative metrics
* The following table shows the quantitative metrics of the model performance