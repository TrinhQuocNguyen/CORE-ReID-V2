
# ################################## //RESNET 18// ################################## 
# ## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2msmt/resnet18/2000_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet18/source_pretraining -b 128 \
                       --num-clusters 2000 --arch resnet18

# # ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet18/2000_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet18/source_pretraining -b 128 \
#                        --num-clusters 2000 --arch resnet18


################################## //RESNET 101// ################################## 
## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet101/2000_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
#                        --num-clusters 2000 --arch resnet101

# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet101/2500_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
#                        --num-clusters 2500 --arch resnet101

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2msmt/resnet101/3000_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
                       --num-clusters 3000 --arch resnet101





# ################################## //RESNET 34// ################################## 
# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet34/2000_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet34/source_pretraining -b 128 \
#                        --num-clusters 2000 --arch resnet34

# ################################## //RESNET 50// ################################## 
# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet50/2000_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet50/source_pretraining -b 128 \
#                        --num-clusters 2000 --arch resnet50

# ################################## //RESNET 152// ################################## 
# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                        --logs-dir logs/cuhk03np2msmt/resnet152/2000_ECAB1_LR \
#                        --initial-weights logs/cuhk03np2market/resnet152/source_pretraining -b 64 \
#                        --num-clusters 2000 --arch resnet152

