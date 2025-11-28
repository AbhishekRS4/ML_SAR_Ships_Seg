# SARShips

## Info
* The project is regarding the segmentation of ships in the processed images generated from the Synthetic Aperture Radar (SAR) data
* This project involves learning some new things like using KServe for deployment of the ML model along with other model optimization strategies

## Instructions to run code
* Run [src/create_label_images.py](src/create_label_images.py) to generate label images from the COCO style format JSON labels
* Run [src/create_pure_bg_label_images.py](src/create_pure_bg_label_images.py) to generate pure background label images

## Additional code optimization info
* The time taken to process labels in JSON format to create label images in a format that is easy to train a ML model
* The optimization used is the ThreadPoolExecutor to generate label images using multiple worker threads. The choice of ThreadPoolExecutor is due to the nature of the task that involes I/O bound operations i.e. saving the images to the disk