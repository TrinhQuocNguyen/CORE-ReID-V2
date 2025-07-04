## step 1 Source-domain pre-training
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds duke -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid  --logs-dir logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --arch resnet101_source

## step 2 Target-domain fine-tuning
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                --logs-dir logs/duke2market_101_all_merge_ECAB/700 \
#                --initial-weights logs/duke2market_101_all_merge_ECAB/source_pretraining \
#                -b 128 --num-clusters 700 --arch resnet101

# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                --logs-dir logs/duke2market_101_all_merge_ECAB/700_ECABX1 \
#                --initial-weights logs/duke2market_101_all_merge_ECAB/source_pretraining \
#                -b 128 --num-clusters 700 --arch resnet101 --resume logs/duke2market_101_all_merge_ECAB/700_ECABX1

# step 1 Source-domain pre-training
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds duke -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                 --logs-dir logs/duke2market_ECAB_BFMN/source_pretraining \
#                 --epochs 150 --eval-step 3 \
#                 -b 128 --arch resnet101_source


CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
               --logs-dir logs/duke2market_ECAB_BFMN/900_ECABX1_LGPR_RandomGrayscale_Kmeanspp \
               --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining \
               --epochs 80 \
               -b 128 --num-clusters 900 --arch resnet101 
               #--resume logs/duke2market_101_all_merge_ECAB/target_fine_tuning

# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                --logs-dir logs/duke2market_101_all_merge_ECAB/700_ECABX_SABX \
#                --initial-weights logs/duke2market_101_all_merge_ECAB/source_pretraining \
#                -b 128 --num-clusters 700 --arch resnet101


## step 3 Evaluate in the target domain
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_700/model_best.pth.tar --num-classes 700 --arch resnet101 


# ## step 2 Target-domain fine-tuning
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_ECAB_BFMN/target_fine_tuning_500 --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --num-clusters 500 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_500/model_best.pth.tar --num-classes 500 --arch resnet101 


# ## step 2 Target-domain fine-tuning
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_ECAB_BFMN/target_fine_tuning_900 --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --num-clusters 900 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_900/model_best.pth.tar --num-classes 900 --arch resnet101 
