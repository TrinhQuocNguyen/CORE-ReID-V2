## step 1 Source-domain pre-training
# Market to Duke
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds market -dt duke --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2duke_101_all_merge/source_pretraining -b 128 --arch resnet101_source #--dropout 0.3

## step 2 Target-domain fine-tuning
# Market to Duke
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt duke --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2duke_101_all_merge/target_fine_tuning --initial-weights logs/market2duke_101_all_merge/source_pretraining -b 128 --num-clusters 900 --arch resnet101

## step 3 Evaluate in the target domain
# Market to Duke
# CUDA_VISIBLE_DEVICES=0,1 python model_save_feature_maps.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/duke2market_101_all_merge/target_fine_tuning/model_best.pth.tar --num-classes 700 --arch resnet101 --batch-size 4
# python model_save_feature_maps.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/cuhk03np2market/target_fine_tuning_900/model_best.pth.tar --num-classes 900 --arch resnet101 --batch-size 1
# python model_save_feature_maps.py -dt cuhk03np --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/market2cuhk03np_ECAB_BFMN/target_fine_tuning_900/model_best.pth.tar --num-classes 900 --arch resnet101 --batch-size 1
# python model_save_feature_maps.py -dt msmt17 --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/market2msmt_ECAB_BFMN/target_fine_tuning_3000/model_best.pth.tar --num-classes 3000 --arch resnet101 --batch-size 1
# python model_save_feature_maps.py -dt msmt17 --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/cuhk03np2msmst_ECAB_BFMN/target_fine_tuning_2500/model_best.pth.tar --num-classes 2500 --arch resnet101 --batch-size 1

# python model_save_feature_maps.py -dt veri --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                                 --resume logs/vehicleid2veri_gan/resnet101/500_ECAB1_LR_000068/model_best.pth.tar \
#                                 --num-classes 500 --arch resnet101 --batch-size 1

# python model_save_feature_maps.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
#                                 --resume logs/cuhk03np2market/resnet101/900_ECAB1_LR/model_best.pth.tar \
#                                 --num-classes 900 --arch resnet101 --batch-size 1

python model_save_feature_maps.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                                --resume logs/cuhk03np2msmt/resnet101/2000_ECAB1_LR/model_best.pth.tar \
                                --num-classes 2000 --arch resnet101 --batch-size 1
