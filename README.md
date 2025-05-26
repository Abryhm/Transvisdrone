.# TransVisDrone
Arxiv Preprint: https://arxiv.org/abs/2210.08423

Major Update: Paper Accepted to 2023 IEEE International Conference on Robotics and Automation (**ICRA**) 🎉 

[Project Page link](https://tusharsangam.github.io/TransVisDrone-project-page/) of original repository. (https://github.com/tusharsangam/TransVisDrone/blob/main/README.md)

 
# Processing NPS Dataset
Download annotations from [Dogfight Github](https://github.com/mwaseema/Drone-Detection?tab=readme-ov-file#annotations) understand the annotation format.
Download the videos from Original [NPS site](https://engineering.purdue.edu/~bouman/UAV_Dataset/)
## Step 1: 
<strong>For the NPS dataset </strong>
Extract all the frames and masks by using the "video_to_frames_and masks.py" .<be>
## Step 2 : 
By using the file "map_masks_images.py," convert the masks into YOLO.txt  format. 
## Step 3 :
Order the names of the frames and labels images by using "reorder_images_labels.py".
## Step 4: 
Rename the images and labels from "000xxx.png" to "00xxx.png" by using the "rename_images_labels.py".

## Step 5 :
Train, Val, Test split followed as in  [dogfight paper](https://arxiv.org/pdf/2103.17242.pdf). (train 0-36, val 37-40, and test 41-50) for all images, labels, and videos. 
Please change the root paths accordingly in [NPS.yaml](./data/NPS.yaml)


# Training NPS,
Please follow whatever parameters are set . 
"python train.py --img 1280 --adam --batch 4 --epochs 80 --data data/NPS_original.yaml --weights yolov5l.pt --hyp data/hyps/hyp.VisDrone_1.yaml --cfg models/yolov5l-xl.yaml --project runs/train/ --name T2 --exist-ok"

In training  refers to 24GB NVIDIA ampere gpu.

# Evaluate NPS results
For validation "python val.py --data data/NPS_original.yaml --weights runs/train/T2/weights/last.pt --batch-size 5 --img 1280 --num-frames 1 --project runs/train/ --name best --task test --exist-ok --save-aot-predictions --save-json-gt" 

For Detect or Inferance use "inference.py" 

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
If you have any questions for this repostry , please feel free to contact us:

[Abdul rehman ]: [phdcs23002@itu.edu.pk](mailto:phdcs23002@itu.edu.pk)


# References
* [yolov5-tph](https://github.com/cv516Buaa/tph-yolov5)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
