## TransVisDrone for NPS Dataset
Arxiv Preprint: https://arxiv.org/abs/2210.08423

Major Update: Paper Accepted to 2023 IEEE International Conference on Robotics and Automation (**ICRA**) 🎉 

[Project Page link](https://tusharsangam.github.io/TransVisDrone-project-page/) of the original repository. (https://github.com/tusharsangam/TransVisDrone/blob/main/README.md)

# Installation

1: git clone https://github.com/Abryhm/Transvisdrone.git

2: conda create --name transvisdrone python==3.9

3: conda activate transvisdrone

4: For the installation, install CUDA and PyTorch "conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge"

5: Install the "pip install -r requirements.txt"


# Processing NPS Dataset
Download annotations from [Dogfight Github](https://github.com/mwaseema/Drone-Detection?tab=readme-ov-file#annotations) understand the annotation format.
Download the videos from the Original [NPS site](https://engineering.purdue.edu/~bouman/UAV_Dataset/)

1. Extract all the frames and masks using the "video_to_frames_and_masks.py".

2. Using the file "map_masks_images.py," convert the masks into YOLO.txt  format. 

3. Order the names of the frames and labels images by using "reorder_images_labels.py".

4. Rename the images and labels from "000xxx.png" to "00xxx.png" by using the "rename_images_labels.py".

5. Train, Val, Test split followed as in  [dogfight paper](https://arxiv.org/pdf/2103.17242.pdf). (train 0-36, val 37-40, and test 41-50) for all images, labels, and videos. 

6.  Please change the root paths accordingly in [NPS.yaml](./data/NPS.yaml)

7. Use the augmentation hyperparameter settings for single Frame "data/hyps/hyp.VisDrone_1.yaml" for three frames "data/hyps/hyp.VisDrone_3.yaml" and for five frames "data/hyps/hyp.VisDrone.yaml" 


# Processing Any Other Dataset

1. Extract all the frames from the video.

2. Convert the annotations into the YOLO.txt format.

3. Make sure the names of images and the labels  are in <strong> "00xxx.png" </strong>  format.

4.  Please change the root paths accordingly in "data/NPS.yaml"

5. Use the augmentation hyperparameter settings for single frame "data/hyps/hyp.VisDrone_1.yaml", for three frames "data/hyps/hyp.VisDrone_3.yaml", and for five frames "data/hyps/hyp.VisDrone.yaml" 


<strong> For training using hyp.VisDrone_1.yaml (i.e., 1 frame) and batch size 4, we need a 24GB GPU such as the NVIDIA GPU 4090. </strong> 

<strong> For training using hyp.VisDrone_3.yaml or hyp.VisDrone.yaml (i.e., 3 or 5 frames) and batch size 4, we need a 48GB GPU </strong> 

<strong> The batch size can be reduced to low resources </strong>



# Training

The training can be done using the following parameters:

"python train.py --img 1280 --adam --batch 4 --epochs 80 --data data/NPS.yaml --weights yolov5l.pt --hyp data/hyps/hyp.VisDrone_1.yaml --cfg models/yolov5l-xl.yaml --project runs/train/ --name T2 --exist-ok"


# Evaluation
For validation "python val.py --data data/NPS.yaml --weights runs/train/T2/weights/last.pt --batch-size 5 --img 1280 --num-frames 1 --project runs/train/ --name best --task test --exist-ok --save-aot-predictions --save-json-gt" 

<strong> To plot the detections, use  "python inference.py --data data/NPS.yaml --task test --weights runs/train/T2/weights/last.pt --img 1280 --conf-thres 0.1" <strong>  

# Citation
If you find our work useful in your research, please consider citing:
``` bash
@INPROCEEDINGS{10161433,
  author={Sangam, Tushar and Dave, Ishan Rajendrakumar and Sultani, Waqas and Shah, Mubarak},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos}, 
  year={2023},
  volume={},
  number={},
  pages={6006-6013},
  keywords={Performance evaluation;Visualization;Image edge detection;Robot vision systems;Transformers;Throughput;Real-time systems},
  doi={10.1109/ICRA48891.2023.10161433}}
```

# Contact
If you have any questions about this repository, please feel free to contact us:

[Abdul rehman ](https://www.linkedin.com/in/abdul-rehman-079348122/): [phdcs23002@itu.edu.pk](mailto:phdcs23002@itu.edu.pk)


# References
* [yolov5-tph](https://github.com/cv516Buaa/tph-yolov5)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
