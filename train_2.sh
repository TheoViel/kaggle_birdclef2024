export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

torchrun --nproc_per_node=4 main_vit_2.py
echo

# torchrun --nproc_per_node=4 main_vit_2.py --model efficientvit_b1
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py --model mixnet_s
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py --model mobilenetv2_100
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py --model mnasnet_100
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py --model tf_efficientnet_b0
# echo

# torchrun --nproc_per_node=4 main_cnn_2.py --model tinynet_b
# echo
