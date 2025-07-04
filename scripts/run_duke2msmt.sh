# # step 1 Source-domain pre-training
# for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds duke -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid  --logs-dir logs/duke2msmt_ECAB_BFMN/source_pretraining -b 128 --arch resnet101_source

# ## step 2 Target-domain fine-tuning
# # for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2msmt_ECAB_BFMN/target_fine_tuning_3000 --initial-weights logs/duke2msmt_ECAB_BFMN/source_pretraining -b 128 --num-clusters 3000 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2msmt_ECAB_BFMN/target_fine_tuning_3000/model_best.pth.tar --num-classes 3000 --arch resnet101 


# ## step 2 Target-domain fine-tuning
# # for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2msmt_ECAB_BFMN/target_fine_tuning_2500 --initial-weights logs/duke2msmt_ECAB_BFMN/source_pretraining -b 128 --num-clusters 2500 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2msmt_ECAB_BFMN/target_fine_tuning_2500/model_best.pth.tar --num-classes 2500 --arch resnet101 


## step 2 Target-domain fine-tuning
# for example, duke-to-msmt17
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2msmt_ECAB_BFMN/target_fine_tuning_3500 --initial-weights logs/duke2msmt_ECAB_BFMN/source_pretraining -b 128 --num-clusters 3500 --arch resnet101

## step 3 Evaluate in the target domain
# for example, duke-to-msmt17
CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2msmt_ECAB_BFMN/target_fine_tuning_3500/model_best.pth.tar --num-classes 3500 --arch resnet101 