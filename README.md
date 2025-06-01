# 4TH PLACE SOLUTION

## Team C-B members
| Name       | GitHub                | LinkedIn               | Zindi                 |
|------------|------------------------|-------------------------|------------------------|
| Durojaiye Abisoye   | https://github.com/DurojaiyeAbisoye  |www.linkedin.com/in/abisoye-durojaiye | https://zindi.africa/users/Bisoye |
| Agbaje Ayomipo |https://github.com/AgbajeAyomipo | https://www.linkedin.com/in/agbaje-ayomipo-6b742423b | https://zindi.africa/users/crossentropy |



## Overview and Objectives

This solution aims to develop an object detection model to identify multiple diseases in cocoa plant images. Our solution is designed to assist subsistence farmers in Africa by enabling disease detection using entry-level smartphones. The objectives include:

- Accurate multi-class disease detection on cocoa plant images.
- Generalization to unseen diseases not present in the training set.
- Efficient deployment and inference on edge devices.

Expected outcomes include increased early detection of plant diseases, reduced crop losses, and minimized pesticide use.

## Folder structure
```
├── README.md  
├── requirements.txt  
├── notebooks/  
│   ├── data-preparation.ipynb  
│   ├── rfdetr-training.ipynb  
│   ├── rfdetr-inference.ipynb  
│   └── wbf.ipynb  
└── results/  
    ├── inference output.csv  
    └── rfdetr_9_epochs_full_linear_best_ema_wbf-max.csv  
```

## Approach
* Model: We make use of [rfdetr](https://github.com/roboflow/rf-detr/tree/develop), a new SOTA real-time object detection model released by [roboflow](https://github.com/roboflow), that supports ONNX deployment natively.
* Data preparation(data-preparation.ipynb):
  *  We split the dataset into train and validation sets using StratifiedKFold with k of 10. We use fold 0 for validation and the other folds for training.
  *  We convert our split dataset into COCO format as required by the rfdetr 
*  Training(rfdetr-training.ipynb): We trained the RfDetr large for 10 epochs using the following configuration
   * epochs: 9
   * batch size: 2
   * grad_accum_steps: 8
   * lr: 1e-4
   * resolution: 448
* Inference(rfdetr-inference.ipynb): RfDetr produces 2 checkpoints: regular weights and EMA-based weights. We made use of the EMA-based weights as those have better performance locally and on the leaderboard. We used a low confidence threshold of 0.02
* Ensembling(wbf.ipynb): We use wbf on the output from the inference notebook. We made use of the [ensemble-boxes framework](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) with the following parameters:
  * conf_type: 'max'
  * skip_box_thr: 0.0001
  * iou_thr: 0.6 

 You can find the full training hparams here:
 ```
Namespace(num_classes=3, grad_accum_steps=8, amp=True, lr=0.0001, lr_encoder=0.00015, batch_size=2,
weight_decay=0.0001, epochs=9, lr_drop=100, clip_max_norm=0.1, lr_vit_layer_decay=0.8, lr_component_decay=0.7,
do_benchmark=False, dropout=0, drop_path=0.0, drop_mode='standard', drop_schedule='constant', cutoff_epoch=0,
pretrained_encoder=None, pretrain_weights='rf-detr-large.pth', pretrain_exclude_keys=None, pretrain_keys_modify_to_load=None, pretrained_distiller=None, encoder='dinov2_windowed_base', vit_encoder_num_layers=12, window_block_indexes=None,
position_embedding='sine', out_feature_indexes=[2, 5, 8, 11], freeze_encoder=False, layer_norm=True, rms_norm=False,
 backbone_lora=False, force_no_pretrain=False, dec_layers=3, dim_feedforward=2048, hidden_dim=384, sa_nheads=12, ca_nheads=24, num_queries=300, group_detr=13, two_stage=True, projector_scale=['P3', 'P5'], lite_refpoint_refine=True, num_select=300,
dec_n_points=4, decoder_norm='LN', bbox_reparam=True, freeze_batch_norm=False, set_cost_class=2, set_cost_bbox=5, set_cost_giou=2,
cls_loss_coef=1.0, bbox_loss_coef=5, giou_loss_coef=2, focal_alpha=0.25, aux_loss=True, sum_group_losses=False,
use_varifocal_loss=False, use_position_supervised_loss=False, ia_bce_loss=True, dataset_file='roboflow', coco_path=None,
dataset_dir='/kaggle/input/amini-cocoa-disease-coco-dataset/amini cocoa disease coco dataset', square_resize_div_64=True,
output_dir='output', dont_save_weights=False, checkpoint_interval=10, seed=42, resume='', start_epoch=0, eval=False, use_ema=True,
ema_decay=0.993, ema_tau=100, num_workers=2, device='cuda', world_size=1, dist_url='env://', sync_bn=True, fp16_eval=False, encoder_only=False, backbone_only=False, resolution=448, use_cls_token=False, multi_scale=True, expanded_scales=True, warmup_epochs=0,
lr_scheduler='step', lr_min_factor=0.0, early_stopping=False, early_stopping_patience=10, early_stopping_min_delta=0.001,
early_stopping_use_ema=False, gradient_checkpointing=False, tensorboard=True, wandb=True, project='Amini cocoa rf-detr',
run='9 epochs from beginning', class_names=[], distributed=False)
 ```
## Metrics
- Train Loss: 6.01317
- Test Loss: 6.74589
- Base Model AP metrics
  - AP50: 0.78086
  - AP50_90: 0.54817
  - AR50_90: 0.43859 
- EMA Model AP metrics
  - AP50: 0.80309
  - AP50_90: 0.57591
  - AR50_90: 0.44415
    
## Runtime
- Training: 8h 24m
- Inference: 21m

## Submission Scores:
  - Public Leaderboard: 0.825360226
  - Private Leaderboard: 0.825496953
