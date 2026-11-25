## Dataset 
The overall directory structure should be:

```
│TGCS/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──.......
```
**ModelNet Dataset:** You can download the processed ModelNet data from [Google Drive](https://drive.google.com/drive/folders/1fAx8Jquh5ES92g1zm2WG6_ozgkwgHhUq?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/), or [Baidu Cloud](https://pan.baidu.com/s/18XL4_HWMlAS_5DUH-T6CjA) (access code 4u1e) and save it in `data/ModelNet/modelnet40_normal_resampled/`. (You can download the official ModelNet from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and process it before training.) Finally, the directory structure should be:

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```

**ModelNet40 few-shot episodes.** Place the episode pickle files in `data/ModelNetFewshot/{way}way_{shot}shot/{fold}.pkl`. Use the same files across methods when comparing accuracy. To generate new random episodes from the processed ModelNet40 files, run `python datasets/generate_few_shot_data.py` from the repository root.

**ScanObjectNN Dataset:** Download the official data from [here](http://103.24.77.34/scanobjectnn) (update: terms of use form required before get the download link, check [here](https://docs.google.com/forms/d/e/1FAIpQLSc6tXkTIBkxIgnjOyK-KKj6s2nLBOzJTQdSdlCDHii0C0bhpw/viewform)) and unzip it into `data/ScanObjectNN`. The directory structure should be:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```


**ShapeNetPart Dataset:** Download the official data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Unzip the file under `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. The directory structure should be

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```
