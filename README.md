# Flow Matching

Pytorch implementation of Flow Matching for image generation on MNIST 

## Setup
```bash
uv venv 
source .venv/bin/activate 
uv pip install -r requirements.txt
```

## Train
```bash
python train.py --data_dir ./data --ckpt_dir ./checkpoints  
```

For additional arguments:
```bash
python train.py -h
```

## Sample
```bash
python sample.py --ckpt_path ./checkpoints/MNIST_16000steps.pt --model ema
```

For additional arguments:
```bash
python sample.py -h 
```

> If you train with different `max_steps`, update the checkpoint path accordingly
