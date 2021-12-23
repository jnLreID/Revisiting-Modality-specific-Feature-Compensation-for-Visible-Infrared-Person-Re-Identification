# Revisiting-Modality-specific-Feature-Compensation-for-Visible-Infrared-Person-Re-Identification

# TSME-Two-Stage Modality Enhancement Network for VI-ReID
Pytorch Code of TSME for Visible-Infrared Person Re-Identification.

## Highlight

The goal of this work is to learn a robust and discriminative compensated features between visible images and infrared images for VI-ReID.

- DSGAN: It can well generate high-quality images by introducing a style transferer between visible grayscale images and infrared images.

- PwIF moduel and IAI module: It enhances the discriminability of the fused feature by constraining the original images and generated paired-images.

### Results on the SYSU-MM01 Dataset
Method |Datasets    | Rank@1  | mAP |
|------| --------      | -----  |  -----  |
| TSME|Single-shot All-Search | ~ 62.65%  | ~ 61.18% |
| TSME|Multi-shot All-Search  | ~ 67.14%  | ~ 54.22% |
| TSME|Single-shot Indoor-Search  | ~ 65.16%  | ~ 72.17% |
| TSME|Multi-shot Indoor-Search   | ~ 76.82%  | ~ 65.72% |

### Results on the RegDB Dataset
Method |Datasets    | Rank@1  | mAP |
|------| --------      | -----  |  -----  |
|TSME|Visible to Infrared| ~ 88.02%  | ~ 79.12% |
|TSME|Infrared to Visible| ~ 86.98%  | ~ 78.23% |

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
  
  - `--resume`: the saved model path. ** Important **(Ours trained model:https://pan.baidu.com/s/1k2nNjYoztD-WpiU5_kFxlQ 
提取码：****)
  
  - `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite the references in your publications if it helps your research:
```
@inproceedings{r19,
	title={Rgb-infrared cross-modality person re-identification},
	author={Wu, Ancong and Zheng, Wei-Shi and Yu, Hong-Xing and Gong, Shaogang and Lai, Jianhuang},
	booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	pages={5380--5389},
	year={2017},
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
