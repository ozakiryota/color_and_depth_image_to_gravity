# color_and_depth_image_to_gravity
## Datasets
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Usage
The following commands are just an example.
### Training
```bash
$ cd ***/color_and_depth_image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 train.py
```
### Validation
```bash
$ cd ***/color_and_depth_image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 infer.py
```
## Citation
If this repository helps your research, please cite the paper below.  
```TeX
@Inproceedings{ozaki2021,
	author = {尾崎亮太 and 黒田洋司}, 
	title = {風景知識を学習するカメラ-LiDAR-DNNによる自己姿勢推定},
	booktitle = {第26回ロボティクスシンポジア予稿集},
	pages = {249--250},
	year = {2021}
}
```
