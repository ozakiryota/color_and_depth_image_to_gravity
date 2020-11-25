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
