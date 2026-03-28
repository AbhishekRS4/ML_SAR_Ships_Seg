# Project To Do List

## To do list for data preparation and visualization generation
- [x] Create a utility script for generating semantic segmentation and instance segmentation label files
- [x] Use ThreadPoolExecutor optimization for multi-processing of label files
- [x] Show a comparison of time speedup with a readme
- [x] Create a utility script for generating visualization of the image, GT label and model predicted label
- [x] Use ProcessPoolExecutor optimization for multi-processing of generation of visualizations

## To do list for Semantic Segmentation Task
- [x] Metrics submodule
- [x] Data handler submodule
- [x] Trainer submodule
- [x] Inference submodule
    - [ ] Eval
    - [x] Inference in normal mode
    - [x] Inference using AOT inductor optimization
    - [x] Inference using ONNX framework
    - [ ] Inference using TensorRT framework
- [x] Model submodule
- [ ] Update readme for the project specific details of the Semantic Segmentation model
    - [x] Data preparation pipeline optimization
    - [x] Visualization pipeline optimization
    - [x] Training pipeline optimization
    - [x] Inference pipeline optimization using AOT inductor
    - [x] Inference pipeline optimization using ONNX
    - [ ] Inference pipeline optimization using TensorRT
    - [x] Model performance metrics
    - [x] Test set model prediction visualizations