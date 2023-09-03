# 1. Vision Transformer in PyTorch introduce
This example shows a simple implementation of [Vision Transformer](https://arxiv.org/abs/2010.11929)<br>
on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.<br>

# 2. 复现步骤
```bash
# pytorch==1.13.0 + cu116; torchvision==1.14.1; numpy==1.23.5也可以
pip3 install -r requirements.txt
python3 main.py
```

# 3. 其它选项
```bash
usage: main.py [-h] [--no-cuda] [--patch-size PATCH_SIZE] [--latent-size LATENT_SIZE] [--n-channels N_CHANNELS] [--num-heads NUM_HEADS] [--num-encoders NUM_ENCODERS]
               [--dropout DROPOUT] [--img-size IMG_SIZE] [--num-classes NUM_CLASSES] [--epochs EPOCHS] [--lr LR] [--weight-decay WEIGHT_DECAY] [--batch-size BATCH_SIZE]

Vision Transformer in PyTorch

options:
  -h, --help            show this help message and exit
  --no-cuda             disables CUDA training
  --patch-size PATCH_SIZE
                        patch size for images (default : 16)
  --latent-size LATENT_SIZE
                        latent size (default : 768)
  --n-channels N_CHANNELS
                        number of channels in images (default : 3 for RGB)
  --num-heads NUM_HEADS
                        (default : 16)
  --num-encoders NUM_ENCODERS
                        number of encoders (default : 12)
  --dropout DROPOUT     dropout value (default : 0.1)
  --img-size IMG_SIZE   image size to be reshaped to (default : 224
  --num-classes NUM_CLASSES
                        number of classes in dataset (default : 10 for CIFAR10)
  --epochs EPOCHS       number of epochs (default : 10)
  --lr LR               base learning rate (default : 0.01)
  --weight-decay WEIGHT_DECAY
                        weight decay value (default : 0.03)
  --batch-size BATCH_SIZE
                        batch size (default : 4)

```