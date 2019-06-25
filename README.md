# ACCM

This is our implementation for the paper:

*Shaoyun Shi, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. [Attention-based Adaptive Model to Unify Warm and Cold Starts Recommendation.](https://dl.acm.org/citation.cfm?id=3271710) 
In CIKM'18.*

**Please cite our paper if you use our codes. Thanks!**

Author: Shaoyun Shi (shisy13 AT gmail.com)

```
@inproceedings{shi2018attention,
  title={Attention-based Adaptive Model to Unify Warm and Cold Starts Recommendation},
  author={Shi, Shaoyun and Zhang, Min and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={127--136},
  year={2018},
  organization={ACM}
}
```



## Environments

Python 3.5.2

Packages: See in [requirements.txt](https://github.com/THUIR/ACCM/blob/master/requirements.txt)

```
tensorflow_gpu==1.4.0
pandas==0.23.1
numpy==1.14.5
tqdm==4.23.4
```



## Datasets

- **ml-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/). The processed ml-100k dataset is in [./dataset](https://github.com/THUIR/ACCM/blob/master/dataset). The codes for processing the data are in [./src/ml-100k.py](https://github.com/THUIR/ACCM/blob/master/src/ml-100k.py).



## Example to run the codes		

```
> cd ACCM
> mkdir model
> cd src

# ACCM with Cold-Sampling
> python CSACCM.py --warm_ratio 0.9

# ACCM without Cold-Sampling
> python CSACCM.py --warm_ratio 1.0
```

> Note that other codes ending with `*Model.py` are inherited by `CSACCM.py`
