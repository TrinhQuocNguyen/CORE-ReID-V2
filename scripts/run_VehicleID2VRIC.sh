## step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds veri -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                          #--logs-dir logs/VeRi2VRIC/resnet101/source -b 128 --arch resnet101_source \
                          #--epochs 2 --iters 1 --eval-step 1

## step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       #--logs-dir logs/VeRi2VRIC/resnet101/fine_900 \
                       #--initial-weights logs/VeRi2VRIC/resnet101/source -b 128 \
                       #--num-clusters 900 --arch resnet101 \
                       #--epochs 2 --iters 1 --eval-step 1


# step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds vehicleid -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                          --logs-dir logs/VehicleID2VRIC/resnet101/source_256 -b 128 --arch resnet101_source \
#                          --height 256 --width 256 \
#                          --epochs 2 --iters 1 --eval-step 1

## step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/VehicleID2VRIC/resnet101/fine_256s_5000c \
                        --initial-weights logs/VehicleID2VRIC/resnet101/source_256 -b 4 \
                        --num-clusters 100 --arch resnet101 \
                        --height 256 --width 256 \
                        --epochs 2 --iters 1 --eval-step 1


# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/VeRi2VRIC/resnet101/fine_256 \
#                         --initial-weights logs/VeRi2VRIC/resnet101/source_256 -b 128 \
#                         --num-clusters 5000 --arch resnet101 \
#                         --height 256 --width 256

# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --resume logs/VeRi2VRIC/resnet101/fine_256/model_best.pth.tar -b 128 \
#                         --num-classes 5000 --arch resnet101 \
#                         --height 256 --width 256
                    
