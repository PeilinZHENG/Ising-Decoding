cd /home/cmcc/Ising-Decoding

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export PREDECODER_TRAIN_EPOCHS=1
export PREDECODER_TRAIN_SAMPLES=8192
export PREDECODER_VAL_SAMPLES=1024
export PREDECODER_TEST_SAMPLES=1024
export PREDECODER_DISABLE_SDR=1
export PREDECODER_LER_FINAL_ONLY=1
export PREDECODER_EVAL_NUM_WORKERS=0
export PREDECODER_SDR_NUM_WORKERS=0
export PREDECODER_INFERENCE_NUM_WORKERS=0
export PREDECODER_TORCH_COMPILE=0

export CUSTOM_DIST_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

python -u -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=1 \
  code/workflows/run.py \
  --config-name=config_public \
  workflow.task=train \
  +exp_tag=ddp2_probe_no_p2p \
  ++load_checkpoint=False \
  hydra.run.dir=/home/cmcc/Ising-Decoding/outputs/ddp2_probe_no_p2p