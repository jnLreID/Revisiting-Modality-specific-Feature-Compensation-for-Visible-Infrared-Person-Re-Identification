# Revisiting-Modality-specific-Feature-Compensation-for-Visible-Infrared-Person-Re-Identification

# TSME
Pytorch Code of TSME for Visible-Infrared Person Re-Identification.

## Highlight

The goal of this work is to learn a robust and discriminative compensated features between visible images and infrared images for VI-ReID.

- DSGAN: It can well generate high-quality generated images by introducing the style transfer between visible grayscale images and infrared images.

- PwIF moduel and IAI module: It enhances the representation of the fused feature by constraining the original images and generated paired-images.

### Results on the SYSU-MM01 Dataset
Method |Datasets    | Rank@1  | mAP |  mINP | 
|------| --------      | -----  |  -----  | ----- |
| AGW [[1](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)]  |#SYSU-MM01 (All-Search)    | ~ 47.50%  | ~ 47.65% | ~ 35.30% | 
| DDAG|#SYSU-MM01 (All-Search)  | ~ 54.75%  | ~ 53.02% | ~39.62% |
| AGW [[1](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)] |#SYSU-MM01 (Indoor-Search)    | ~ 54.17% | ~ 62.97% | ~ 59.23%| 
| DDAG|#SYSU-MM01 (Indoor-Search)  | ~ 61.02% | ~ 67.98% | ~ 62.61%|

*The code has been tested in Python 3.7, PyTorch=1.0. Both of these two datasets may have some fluctuation due to random spliting

### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be requested via sending me an email (mangye16@gmail.com). 
  
- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` [link](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/blob/master/pre_process_sysu.py) in to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training.
  Train a model by
  ```bash
python train.py --dataset sysu --lr 0.1 --gpu number
```

  - `--dataset`: which dataset "sysu" or "regdb".

  - `--lr`: initial learning rate.
    
  - `--gpu`:  which gpu to run.

You may need manually define the data path first.


### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by 
  ```bash
python test.py --dataset sysu --mode all --gpu number --resume 'model_path' 
```
  - `--dataset`: which dataset "sysu" or "regdb".
  
  - `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).
  
  - `--trial`: testing trial (only for RegDB dataset).
  
  - `--resume`: the saved model path. ** Important **
  
  - `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite the references in your publications if it helps your research:
```
@inproceedings{eccv20ddag,
  title={Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification},
  author={Ye, Mang and Shen, Jianbing and Crandall, David J. and Shao, Ling and Luo, Jiebo},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020},
}
```

```
@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```

Contact: jianan_liu@stu.xidian.edu.cn
