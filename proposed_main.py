from __future__ import print_function
import argparse
import os
import GPUtil

## Module
from dataLoader import get_datapath, DataSegmentationLoader
from proposed_utils import *
import proposed_models
import proposed_train

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices


model_dict = {"Unet": 0, "Shallow_Unet": 1, "ResUnet": 2, "ResUnet_plusplus": 3}

parser = argparse.ArgumentParser(description='Brain Tumor Segmentation')
parser.add_argument('--mode', default=3, type=int,
                    help='perturbation magnitude')
parser.add_argument('--input_channel', default=32, type=int,
                    help='perturbation magnitude')
parser.add_argument('--lr', default=1e-5, type=int,
                    help='perturbation magnitude')
parser.add_argument('--batch_size', default=5, type=int,
                    help='perturbation magnitude')
parser.add_argument('--decay', default=0.98, type=int,
                    help='perturbation magnitude')
parser.add_argument('--epochs', default=50, type=int,
                    help='perturbation magnitude')
parser.add_argument('--nfold', default=5, type=int,
                    help='perturbation magnitude')
parser.set_defaults(argument=True)

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def main():
    # Import Data
    global args
    args = parser.parse_args()

    # Use GPU
    import torch.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {device}.')
    else:
        print(f'CUDA is not available. Your device is {device}. It can take long time training in CPU.')

    # Fix Seed
    random_state = 42
    seed_everything(random_state)

    # Dataload
    image, mask = get_datapath('/DataCommon2/ksoh/perceptron_course/Brain_tumor_segmentation2/data/', random_state)

    dataloader = DataSegmentationLoader(image, mask)
    model = proposed_models.proposed_net(ch=args.input_channel, mode=args.mode).build_model()
    proposed_train.train(dataloader, model, args.mode, args.lr, args.batch_size, args.decay, args.epochs, args.nfold)

if __name__ == '__main__':
    main()