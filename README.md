# CMSA
Contrast-driven incremental Multi-view clustering with Semantic distillation and Adaptive graph fusion (CMSA)


## 1.Requirements

pytorch==2.2.1

numpy==1.26.4

scikit-learn==1.4.1

## 2.Datasets
The Prokaryotic datasets is placed in. The others dataset could be downloaded from [cloud](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ). key: data   Source of the dataset：[GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering] (https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.pdf)

## 3.Usage
You need to create the project and then place the code into it. After that, switch the data set and run it.


```bash
Epoch 47 Loss:15.420288
Epoch 48 Loss:15.431067
Epoch 49 Loss:15.417261
Epoch 50 Loss:15.436375
---------train over---------
Clustering results:
ACC = 0.7278 NMI = 0.4918 PUR=0.8457 ARI = 4544
Saving model...
```
## 3.Citation
After setting up the environment as described in the README, you can run the code.

This work is inspired by:

MVCformer: A transformer-based multi-view clustering method (GitHub)

GCFAgg: Global and Cross-view Feature Aggregation for Multi-view Clustering (GitHub)

I would like to express my respect for these outstanding research works.
