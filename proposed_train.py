import torch
from proposed_utils import *
##Cross Validation
from sklearn.model_selection import KFold
import tensorflow.keras as keras
import tensorflow as tf
import os
import tqdm

# main
def train(dataloader, model, mode, lr, batch_size, decay, num_epochs, n_folds, workspace):
    results = {}
    folds = KFold(n_splits = n_folds)

    fff = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    # workspace = workspace + "/proposed_model" + "/" + "mode%d" % mode +"_1e-6_dice"
    workspace = workspace + "/proposed_model" + "/" + "mode000" + "_1e-5_dice"

    # KFold Cross Validation
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(dataloader)):
        print("File path with the name: %s" % workspace)
        if not os.path.isdir(os.path.join(workspace, fff[fold_])): os.makedirs(os.path.join(workspace, fff[fold_]))

        print("fold n°{}".format(fold_+1))
        ##Split by folder and load by dataLoader
        train_subsampler = torch.utils.data.SubsetRandomSampler(trn_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size, sampler=train_subsampler, drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size, sampler=valid_subsampler, drop_last=True)

        # Initialize Model
        train_vars = []
        train_vars += model.trainable_variables

        # Initialize optimizer
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=len(trn_idx) // batch_size, decay_rate=decay, staircase=True)
        optim = keras.optimizers.Adam(lr_schedule)

        # Model selection
        best_iou = 0.0
        for epoch in tqdm.trange(num_epochs):
            train_loss, train_iou = [], []
            valid_loss, valid_iou = [], []

            for batch_idx, (features,targets) in enumerate(train_dataloader):
                features, targets = tf.constant(features), tf.constant(targets)

                with tf.GradientTape() as tape:
                    logits = model({"in": features}, training=True)["out"]
                    dice_loss = DiceLoss(logits, targets)
                    # bce_loss = K.mean(tf.keras.losses.binary_crossentropy(logits, targets))
                    total_loss = dice_loss

                grads = tape.gradient(total_loss, train_vars)
                optim.apply_gradients(zip(grads, train_vars))

                iou = iou_score(targets, logits)*100.0
                train_loss.append(total_loss)
                train_iou.append(iou)

                if not batch_idx % 80:
                    print ('Epoch: %03d/%03d | Batch %03d/%03d | Train Loss: %.4f | Train IoU: %.4f%% '
                           %(epoch+1, num_epochs, batch_idx,
                             len(train_dataloader),
                             np.mean(train_loss),
                             np.mean(train_iou))
                          )

            ##Valid
            for batch_idx, (features,targets) in enumerate(valid_dataloader):
                features, targets = tf.constant(features), tf.constant(targets)

                logits = model({"in": features}, training=False)["out"]
                dice_loss = DiceLoss(logits, targets)
                # bce_loss = tf.keras.losses.binary_crossentropy(logits, targets)
                total_loss = dice_loss
                iou = iou_score(targets, logits)*100.0

                ### LOGGING
                valid_loss.append(total_loss)
                valid_iou.append(iou)
            print('Epoch: %03d/%03d |  Valid Loss: %.4f | Valid IoU: %.4f%%' % (
                  epoch+1, num_epochs,
                  np.mean(valid_loss),
                  np.mean(valid_iou)))

            # model save
            if best_iou <= np.mean(valid_iou):
                best_iou = np.mean(valid_iou)
                model.save(os.path.join(workspace, fff[fold_]) + "/epoch%03d_model" % epoch)
                results[fold_+1] = np.mean(valid_iou)

    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

