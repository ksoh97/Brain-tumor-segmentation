from __future__ import print_function
import argparse
import os
import GPUtil

## Module
from dataLoader import get_datapath, DataSegmentationLoader
from sklearn.model_selection import KFold
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
parser.add_argument('--mode', default=0, type=int,
                    help='perturbation magnitude')
parser.add_argument('--trn_vs_eval', default=True, type=bool,
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
parser.add_argument('--workspace', default="/DataCommon2/ksoh/perceptron_course/Brain_tumor_segmentation2", type=str,
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
    if args.trn_vs_eval:
        image, mask = get_datapath(args.workspace + '/data/', random_state)
        dataloader = DataSegmentationLoader(image, mask)
        model = proposed_models.proposed_net(ch=args.input_channel, mode=args.mode).build_model()
        proposed_train.train(dataloader, model, args.mode, args.lr, args.batch_size, args.decay,
                             args.epochs, args.nfold, args.workspace)

    else:
        file_name = "mode%d_1e-5_dice" % args.mode  # 수정
        image, mask = get_datapath(args.workspace + '/Testdata/', random_state)
        dataloader = DataSegmentationLoader(image, mask)
        model = proposed_models.proposed_net(ch=args.input_channel, mode=args.mode).build_model()

        results = {}
        workspace = args.workspace + "/proposed_model"
        model_path = os.path.join(workspace, file_name)

        from sklearn.model_selection import KFold
        folds = KFold(n_splits=args.nfold)

        # KFold Cross Validation
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(dataloader)):
            model.load_weights(model_path + "/fold%d" % (fold_ + 1) + "/best_model/variables/variables")
            print("fold n°{}".format(fold_ + 1))

            tst_idx = np.sort(np.append(trn_idx, val_idx))
            test_subsampler = torch.utils.data.SubsetRandomSampler(tst_idx)
            test_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=args.batch_size, sampler=test_subsampler,
                                                           drop_last=True)
            test_iou = []
            for batch_idx, (features, targets) in enumerate(test_dataloader):
                features, targets = tf.constant(features), tf.constant(targets)

                logits = model({"in": features}, training=False)["out"]
                iou = iou_score(targets, logits) * 100.0
                test_iou.append(iou)

            results[fold_ + 1] = np.mean(test_iou)

        # Print fold results
        print(f'\nTEST PERFORMANCE RESULTS FOR {args.nfold} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum / len(results.items())} %')

if __name__ == '__main__':
    main()