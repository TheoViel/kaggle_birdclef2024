export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

# torchrun --nproc_per_node=4 main_vit.py --model efficientvit_b0
# echo

# torchrun --nproc_per_node=4 main_vit.py --model efficientvit_b1
# echo

torchrun --nproc_per_node=4 main_vit.py --model efficientvit_m3
echo

# torchrun --nproc_per_node=4 main_cnn.py --model mnasnet_100
# echo

# torchrun --nproc_per_node=4 main_cnn.py --model tf_efficientnet_b0
# echo

# echo
# torchrun --nproc_per_node=4 main_cnn.py --model tf_mobilenetv3_large_minimal_100

# echo
# torchrun --nproc_per_node=4 main_cnn.py --model efficientvit_m3
