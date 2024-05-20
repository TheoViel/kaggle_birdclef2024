export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

torchrun --nproc_per_node=4 main_vit_2.py --model efficientvit_b0
echo

torchrun --nproc_per_node=4 main_cnn_2.py --model mnasnet_100
echo

torchrun --nproc_per_node=4 main_vit_2.py --model efficientvit_m3
echo
