# ################################## //RESNET 101// ################################## 
## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri/resnet101/source_pretraining -b 128 --arch resnet101_source \
#                         --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet101/200_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet101/source_pretraining -b 128 \
#                        --num-clusters 200 --arch resnet101 \
#                        --height 256 --width 256

# Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet101/500_EXPERIMENT \
#                        --initial-weights logs/vehicleid2veri/resnet101/source_pretraining -b 128 \
#                        --num-clusters 500 --arch resnet101 \
#                        --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet101/700_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet101/source_pretraining -b 128 \
#                        --num-clusters 700 --arch resnet101 \
#                        --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet101/900_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet101/source_pretraining -b 128 \
#                        --num-clusters 900 --arch resnet101 \
#                        --height 256 --width 256

# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                          --resume logs/vehicleid2veri/resnet34/700_ECAB1_LR/model_best.pth.tar -b 128 \
#                          --num-classes 700 --arch resnet34 \
#                          --height 256 --width 256 

# ################################## //RESNET 50// ################################## 
# ## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri/resnet50/source_pretraining -b 128 --arch resnet50_source \
#                         --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet50/300_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet50/source_pretraining -b 128 \
#                        --num-clusters 300 --arch resnet50 \
#                        --height 256 --width 256


# ################################## //RESNET 152// ################################## 
# ## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri/resnet152/source_pretraining -b 64 --arch resnet152_source \
#                         --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet152/300_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet152/source_pretraining -b 64 \
#                        --num-clusters 300 --arch resnet152 \
#                        --height 256 --width 256

# ################################## //RESNET 18// ################################## 
# ## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri/resnet18/source_pretraining -b 128 --arch resnet18_source \
#                         --height 256 --width 256

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri/resnet18/300_ECAB1_LR \
#                        --initial-weights logs/vehicleid2veri/resnet18/source_pretraining -b 128 \
#                        --num-clusters 300 --arch resnet18 \
#                        --height 256 --width 256

# ################################## //RESNET 34// ################################## 
# ## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri/resnet34/source_pretraining -b 128 --arch resnet34_source \
#                         --height 256 --width 256

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/vehicleid2veri/resnet34/700_ECAB1_LR \
                       --initial-weights logs/vehicleid2veri/resnet34/source_pretraining -b 128 \
                       --num-clusters 700 --arch resnet34 \
                       --height 256 --width 256