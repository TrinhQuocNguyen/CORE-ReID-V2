## step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds vric -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                         --logs-dir logs/VRIC2VeRi/resnet101/source_256 -b 128 --arch resnet101_source \
                         --height 256 --width 256


CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds veri -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                         --logs-dir logs/VeRi2VRIC/resnet101/source_256 -b 128 --arch resnet101_source \
                         --height 256 --width 256

# ## step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/VRIC2VeRi/resnet101/fine_256 \
#                         --initial-weights logs/VRIC2VeRi/resnet101/source_256 -b 128 \
#                         --num-clusters 5000 --arch resnet101 \
#                         --height 256 --width 256

# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --logs-dir logs/VRIC2VeRi/resnet101/fine_256 \
#                         --initial-weights logs/VRIC2VeRi/resnet101/source_256 -b 128 \
#                         --num-clusters 5000 --arch resnet101 \
#                         --height 256 --width 256

# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt vric --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                         --resume logs/VRIC2VeRi/resnet101/fine_256/model_best.pth.tar -b 128 \
#                         --num-classes 5000 --arch resnet101 \
#                         --height 256 --width 256
                    
