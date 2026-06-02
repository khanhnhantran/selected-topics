import argparse


def gpus_arg(value):
    """Accept either a GPU count ("3") or a list of GPU IDs ("3,5,7")."""
    if ',' in value:
        return [int(v) for v in value.split(',') if v.strip() != '']
    return int(value)


parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

# Training Parameters
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=2,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--de_type', nargs='+', default=['derain', 'desnow'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers.')

# Data Parameters
parser.add_argument('--data_file_dir', type=str, default='data/hw4_realse_dataset/',  help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/hw4_realse_dataset/',
                    help='where training images of deraining saves.')
parser.add_argument('--desnow_dir', type=str, default='data/hw4_realse_dataset/',
                    help='where snowing images of desnowing saves.')
parser.add_argument('--num_aug', type=int, default=120)

# Save dir
parser.add_argument("--wblogger",type=str,default="hw4",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--ckpt_name",type=str, default="best_rainsnow_edge")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume training from (restores epoch, optimizer, scheduler).")
parser.add_argument("--init-from", dest="init_from", type=str, default=None,
                    help="Path to a checkpoint to initialize net.* weights from. "
                         "Unlike --resume, this does NOT restore epoch / optimizer / "
                         "scheduler / EMA state — use for fine-tuning with a fresh schedule.")
parser.add_argument("--wandb_name",type=str, default="RainySnow_edge")
parser.add_argument('--output_path', type=str, default="output/", help='output save path')

# Num of GPUs
parser.add_argument("--num_gpus",type=gpus_arg,default=1,
                    help="Number of GPUs (e.g. 3) or comma-separated GPU IDs (e.g. 3,5,7) to use for training")

options = parser.parse_args()
