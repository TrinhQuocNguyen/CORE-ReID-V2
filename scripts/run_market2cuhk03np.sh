## step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds market -dt cuhk03np --data-dir /home/ccvn/Workspace/trinh/data/reid  --logs-dir logs/market2cuhk03np_ECAB_BFMN/source_pretraining -b 128 --arch resnet101_source

# ## step 2 Target-domain fine-tuning
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt cuhk03np --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2cuhk03np_ECAB_BFMN/target_fine_tuning_900_NoECAB --initial-weights logs/market2cuhk03np_ECAB_BFMN/source_pretraining -b 128 --num-clusters 900 --arch resnet101

# ## step 3 Evaluate in the target domain
# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt cuhk03np --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/market2cuhk03np_ECAB_BFMN/target_fine_tuning_900_NoECAB/model_best.pth.tar --num-classes 900 --arch resnet101 

## step 2 Target-domain fine-tuning
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt cuhk03np --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                     --logs-dir logs/market2cuhk03np_ECAB_BFMN/900_ECABX_SABX_RandomGrayscale \
#                     --initial-weights logs/market2cuhk03np_ECAB_BFMN/source_pretraining \
#                     -b 128 --num-clusters 900 --arch resnet101

## step 3 Evaluate in the target domain
# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt cuhk03np --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/market2cuhk03np_ECAB_BFMN/target_fine_tuning_900_NoECAB/model_best.pth.tar --num-classes 900 --arch resnet101 

## step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds market -dt cuhk03np --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                           --logs-dir logs/market2cuhk03np/resnet18/source_pretraining -b 128 --arch resnet18_source \
#                           --epochs 2 --iters 2 --eval-step 1

## step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt cuhk03np --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/market2cuhk03np/resnet18/900_ECAB1_LR \
                       --initial-weights logs/market2cuhk03np/resnet18/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet18 \
                       --epochs 2 --iters 2 --eval-step 1