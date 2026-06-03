#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eti_exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-a100

cd /mnt/research/j.farias/mkang2/IGLU-Activation 

COUNT=0
FAILED=0

COUNT=$((COUNT + 1)) 

output_dir="./Output/ETI/ResNet20_NoBatchNorm_IGLU_0.1_s42"

echo "[$COUNT] Resnet-20"

python vision_main.py \
    --activation zilu \
    --sigma 0.1 \
    --inplace \
    --model resnet20 \
    --dataset cifar100 \
    --augment \
    --data_path ./Data \
    --batch_size 128 \
    --num_epochs 200 \
    --use_amp \
    --clip_grad_norm 1.0 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 1e-2 \
    --lr 1e-3 \
    --scheduler cosine \
    --device cuda \
    --seed 42 \
    --output_dir $output_dir \
    --num_workers 12 \
    --pin_memory \
    --measure_tail_index

# Check if experiment succeeded
if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


COUNT=$((COUNT + 1)) 

output_dir="./Output/ETI/VIT-Tiny_NoLayerNorm_IGLU_5.0_s42"

echo "[$COUNT]"

python vision_main.py \
    --activation zilu \
    --sigma 5.0 \
    --inplace \
    --model vit-tiny \
    --dataset cifar100 \
    --augment \
    --resize 224 \
    --data_path ./Data \
    --batch_size 128 \
    --num_epochs 200 \
    --use_amp \
    --clip_grad_norm 1.0 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 1e-2 \
    --lr 1e-3 \
    --scheduler cosine \
    --device cuda \
    --seed 42 \
    --output_dir $output_dir \
    --num_workers 12 \
    --pin_memory \
    --measure_tail_index

# Check if experiment succeeded
if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1)) 

output_dir="./Output/ETI/GPT2_NoLayerNorm_IGLU_5.0_s42" 

echo "[$COUNT]"

python language_main.py \
    --vocab_size 50257 \
    --max_seq_length 1024 \
    --embedding_dim 768 \
    --num_attention_heads 12 \
    --num_layers 12 \
    --activation zilu \
    --sigma 5.0 \
    --inplace \
    --dataset wikitext103 \
    --use_amp \
    --data_path ./Data \
    --batch_size 16 \
    --num_epochs 20 \
    --clip_grad_norm 1.0 \
    --optimizer adamw \
    --weight_decay 0.1 \
    --lr 6e-4 \
    --scheduler linear \
    --device cuda \
    --seed 42 \
    --output_dir $output_dir \
    --num_workers 12 \
    --pin_memory \
    --measure_tail_index

# Check if experiment succeeded
if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


