# ################################## //RESNET 101// ################################## 
# Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri_gan/resnet101/source_pretraining -b 128 --arch resnet101_source \
#                         --height 256 --width 256 --epochs 552


# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri_gan/resnet101/source_pretraining -b 128 --arch resnet101_source \
#                         --height 256 --width 256 --epochs 552 --resume logs/vehicleid2veri_gan/resnet101/source_pretraining/model_best.pth.tar --evaluate 


## Step 3 - Test
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --resume logs/vehicleid2veri_gan/resnet101/700_ECAB1_LR \
#                         --num-classes 700 --arch resnet101 \
#                         --height 256 --width 256

## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet101/300_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet101/source_pretraining -b 128 \
#                        --num-clusters 300 --arch resnet101 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068

# ################################## //RESNET 18// ################################## 
# Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri_gan/resnet18/source_pretraining -b 128 --arch resnet18_source \
#                         --height 256 --width 256 --epochs 552


## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet18/500_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet18/source_pretraining -b 128 \
#                        --num-clusters 500 --arch resnet18 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068


# ################################## //RESNET 34// ################################## 
# Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri_gan/resnet34/source_pretraining -b 128 --arch resnet34_source \
#                         --height 256 --width 256 --epochs 552


# # Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet34/500_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet34/source_pretraining -b 128 \
#                        --num-clusters 500 --arch resnet34 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068

# ################################## //RESNET 152// ################################## 
# Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/vehicleid2veri_gan/resnet152/source_pretraining -b 64 --arch resnet152_source \
                        --height 256 --width 256 --epochs 552


## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/vehicleid2veri_gan/resnet152/500_ECAB1_LR_000068 \
                       --initial-weights logs/vehicleid2veri_gan/resnet152/source_pretraining -b 64 \
                       --num-clusters 500 --arch resnet152 \
                       --height 256 --width 256 --epochs 15 --lr 0.000068


# ################################## //RESNET 50// ################################## 
# Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds vehicleid -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/vehicleid2veri_gan/resnet50/source_pretraining -b 128 --arch resnet50_source \
#                         --height 256 --width 256 --epochs 552


# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet50/500_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet50/source_pretraining -b 128 \
#                        --num-clusters 500 --arch resnet50 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet50/700_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet50/source_pretraining -b 128 \
#                        --num-clusters 700 --arch resnet50 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/vehicleid2veri_gan/resnet50/900_ECAB1_LR_000068 \
#                        --initial-weights logs/vehicleid2veri_gan/resnet50/source_pretraining -b 128 \
#                        --num-clusters 900 --arch resnet50 \
#                        --height 256 --width 256 --epochs 15 --lr 0.000068