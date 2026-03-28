# Unified training entrypoint for SSAMBA (AMBAModel) and SSAST (ASTModel).
# Select the model with --model amba or --model ssast.

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask
import wandb

print(
    "I am process %s, running on %s: starting (%s)"
    % (os.getpid(), os.uname()[1], time.asctime())
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--model",
    type=str,
    default="amba",
    choices=["amba", "ssast"],
    help="model architecture to use",
)

# Data args
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default="", help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

# Training args
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, metavar="LR", help="initial learning rate")
parser.add_argument("--warmup", help="if use warmup learning rate scheduler", type=ast.literal_eval, default="True")
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument("-b", "--batch-size", default=12, type=int, metavar="N", help="mini-batch size")
parser.add_argument("-w", "--num-workers", default=16, type=int, metavar="NW", help="# of workers for dataloading")
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--adaptschedule", help="if use adaptive scheduler", type=ast.literal_eval, default="False")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument("--save_model", help="save the models or not", type=ast.literal_eval)

# Augmentation args
parser.add_argument("--freqm", help="frequency mask max length", type=int, default=0)
parser.add_argument("--timem", help="time mask max length", type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--noise", help="if augment noise in finetuning", type=ast.literal_eval)

# Patch args
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument("--model_size", help="the size of the model", type=str, default="base")

# Task args
parser.add_argument(
    "--task",
    type=str,
    default="ft_cls",
    help="pretraining or fine-tuning task",
    choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"],
)
parser.add_argument("--mask_patch", help="how many patches to mask (ssl pretraining only)", type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clustering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, iterations per validation")

# Fine-tuning args
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained model path")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="number of steps to decrease lr in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="learning rate decay ratio in finetuning")
parser.add_argument("--wa", help="if do weight averaging in finetuning", type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning", choices=["BCE", "CE"])
parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases")

# Vision Mamba args (used only when --model amba)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--embed_dim", type=int, default=768)
parser.add_argument("--depth", type=int, default=24)
parser.add_argument("--rms_norm", type=str, choices=["true", "false"], default="true")
parser.add_argument("--residual_in_fp32", type=str, choices=["true", "false"], default="true")
parser.add_argument("--fused_add_norm", type=str, choices=["true", "false"], default="true")
parser.add_argument("--if_rope", type=str, choices=["true", "false"], default="false")
parser.add_argument("--if_rope_residual", type=str, choices=["true", "false"], default="false")
parser.add_argument("--bimamba_type", type=str, default="v2")
parser.add_argument("--drop_path_rate", type=float, default=0.1)
parser.add_argument("--stride", type=int, default=16)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--drop_rate", type=float, default=0.0)
parser.add_argument("--norm_epsilon", type=float, default=1e-5)
parser.add_argument("--if_bidirectional", type=str, choices=["true", "false"], default="true")
parser.add_argument("--final_pool_type", type=str, default="none")
parser.add_argument("--if_abs_pos_embed", type=str, choices=["true", "false"], default="false")
parser.add_argument("--if_bimamba", type=str, choices=["true", "false"], default="true")
parser.add_argument("--if_cls_token", type=str, choices=["true", "false"], default="false")
parser.add_argument("--if_devide_out", type=str, choices=["true", "false"], default="false")
parser.add_argument("--use_double_cls_token", type=str, choices=["true", "false"], default="false")
parser.add_argument("--use_middle_cls_token", type=str, choices=["true", "false"], default="false")

args = parser.parse_args()

# Convert string booleans for Mamba args
for attr in [
    "rms_norm", "residual_in_fp32", "fused_add_norm", "if_rope", "if_rope_residual",
    "if_bidirectional", "if_abs_pos_embed", "if_bimamba", "if_cls_token",
    "if_devide_out", "use_double_cls_token", "use_middle_cls_token",
]:
    setattr(args, attr, getattr(args, attr) == "true")

if args.use_wandb:
    prefix = "amba" if args.model == "amba" else "ssast"
    project_name = f"{prefix}_final"
    if args.dataset == "esc50":
        project_name = f"{prefix}_esc"
    elif args.dataset == "audioset":
        project_name = f"{prefix}_as"
    wandb.init(project=project_name, config=args)

audio_conf = {
    "num_mel_bins": args.num_mel_bins,
    "target_length": args.target_length,
    "freqm": args.freqm,
    "timem": args.timem,
    "mixup": args.mixup,
    "dataset": args.dataset,
    "mode": "train",
    "mean": args.dataset_mean,
    "std": args.dataset_std,
    "noise": args.noise,
}

val_audio_conf = {
    "num_mel_bins": args.num_mel_bins,
    "target_length": args.target_length,
    "freqm": 0,
    "timem": 0,
    "mixup": 0,
    "dataset": args.dataset,
    "mode": "evaluation",
    "mean": args.dataset_mean,
    "std": args.dataset_std,
    "noise": False,
}

if args.bal == "bal":
    print("balanced sampler is being used")
    samples_weight = np.loadtxt(args.data_train[:-5] + "_weight.csv", delimiter=",")
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
else:
    print("balanced sampler is not used")
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

dataset = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(train_loader)}")

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=False,
)

print(
    "Now train with {:s} with {:d} training samples, evaluate with {:d} samples".format(
        args.dataset, len(train_loader.dataset), len(val_loader.dataset)
    )
)

if args.model == "amba":
    from models import AMBAModel

    vision_mamba_config = {
        "img_size": (args.num_mel_bins, args.target_length),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "channels": args.channels,
        "num_classes": args.num_classes,
        "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate,
        "norm_epsilon": args.norm_epsilon,
        "rms_norm": args.rms_norm,
        "residual_in_fp32": args.residual_in_fp32,
        "fused_add_norm": args.fused_add_norm,
        "if_rope": args.if_rope,
        "if_rope_residual": args.if_rope_residual,
        "bimamba_type": args.bimamba_type,
        "if_bidirectional": args.if_bidirectional,
        "final_pool_type": args.final_pool_type,
        "if_abs_pos_embed": args.if_abs_pos_embed,
        "if_bimamba": args.if_bimamba,
        "if_cls_token": args.if_cls_token,
        "if_devide_out": args.if_devide_out,
        "use_double_cls_token": args.use_double_cls_token,
        "use_middle_cls_token": args.use_middle_cls_token,
    }

    if "pretrain" in args.task:
        cluster = args.num_mel_bins != args.fshape
        print(
            "num_mel_bins {:d} and fshape {:d} are {:s}, {:s} cluster masking.".format(
                args.num_mel_bins, args.fshape,
                "different" if cluster else "same",
                "using" if cluster else "not using",
            )
        )
        audio_model = AMBAModel(
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fshape, tstride=args.tshape,
            input_fdim=args.num_mel_bins, input_tdim=args.target_length,
            model_size=args.model_size, pretrain_stage=True,
            vision_mamba_config=vision_mamba_config,
        )
    else:
        audio_model = AMBAModel(
            label_dim=args.n_class,
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fstride, tstride=args.tstride,
            input_fdim=args.num_mel_bins, input_tdim=args.target_length,
            model_size=args.model_size, pretrain_stage=False,
            load_pretrained_mdl_path=args.pretrained_mdl_path,
            vision_mamba_config=vision_mamba_config,
        )

else:  # ssast
    from models import ASTModel

    if "pretrain" in args.task:
        cluster = args.num_mel_bins != args.fshape
        print(
            "num_mel_bins {:d} and fshape {:d} are {:s}, {:s} cluster masking.".format(
                args.num_mel_bins, args.fshape,
                "different" if cluster else "same",
                "using" if cluster else "not using",
            )
        )
        audio_model = ASTModel(
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fshape, tstride=args.tshape,
            input_fdim=args.num_mel_bins, input_tdim=args.target_length,
            model_size=args.model_size, pretrain_stage=True,
        )
    else:
        audio_model = ASTModel(
            label_dim=args.n_class,
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fstride, tstride=args.tstride,
            input_fdim=args.num_mel_bins, input_tdim=args.target_length,
            model_size=args.model_size, pretrain_stage=False,
            load_pretrained_mdl_path=args.pretrained_mdl_path,
        )

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
if not os.path.exists("%s/models" % args.exp_dir):
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

if "pretrain" not in args.task:
    print("Now starting fine-tuning for {:d} epochs".format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)
else:
    print("Now starting self-supervised pretraining for {:d} epochs".format(args.n_epochs))
    trainmask(audio_model, train_loader, val_loader, args)

if args.data_eval is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + "/models/best_audio_model.pth", map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    stats, _ = validate(audio_model, val_loader, args, "valid_set")
    val_acc = stats[0]["acc"]
    val_mAUC = np.mean([stat["auc"] for stat in stats])
    print("---------------evaluate on the validation set---------------")
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    stats, _ = validate(audio_model, eval_loader, args, "eval_set")
    eval_acc = stats[0]["acc"]
    eval_mAUC = np.mean([stat["auc"] for stat in stats])
    print("---------------evaluate on the test set---------------")
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    if args.use_wandb:
        wandb.log({"val_accuracy": val_acc, "val_mAUC": val_mAUC})
    np.savetxt(
        args.exp_dir + "/eval_result.csv",
        [val_acc, val_mAUC, eval_acc, eval_mAUC],
    )
