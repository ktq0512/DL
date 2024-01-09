import os
import argparse
import sys
import logging

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch import nn
import torch


from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.TFMF_TGR import TMFModel

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser = TMFModel.init_args(parser)

# distributed training ways : 1 - lightning, 2 -nn.DataParallel, 3 -nn.DistributedDataParallel 4 -Horovod 

def main():
    args = parser.parse_args()
    # args.reduce_dataset_size 裁剪验证集，不写或者为0就是整个验证集
    # args.reduce_dataset_size = 50
    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    
    val_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers, # PyTorch provides an easy switch to perform multi-process data loading
        collate_fn=collate_fn_dict, # A custom collate_fn can be used to customize collation, convert the list of dictionaries to the dictionary of lists 
        pin_memory=True # For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
    )
    # 整个训练集长度为199908，args.reduce_dataset_size裁剪训练集，不写或者为0就是整个训练集
    # args.reduce_dataset_size = 66636
    # args.reduce_dataset_size = 50
    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre, args)
    print(len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False, # n multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.
        shuffle=True
    )
    # Save the model periodically by monitoring a quantity. Every metric logged with log() or log_dict() in LightningModule is a candidate for the monitor key.
    checkpoint_callback = pl.callbacks.ModelCheckpoint( 
        filename="{epoch}-{loss_val_epoch:.4f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=-1,
    )

    
    model = TMFModel(args)

    # # 读取预训练模型
    # checkpoint = torch.load('./lightning_logs/version_74/checkpoints/epoch=63-ade_val=1.66-fde_val=3.53.ckpt')
    # state_dict = checkpoint['state_dict']
    # pretrain_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder_transformer')}
    # model = TMFModel(args)
    # model_dict = model.state_dict()
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)
    
    # 直接训练
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        gpus=args.gpus,
        weights_save_path=None,
        max_epochs=args.num_epochs
    )

    # # 训练中断，需要恢复
    # trainer = pl.Trainer(
    #     default_root_dir=log_dir,
    #     callbacks=[checkpoint_callback],
    #     gpus=args.gpus,
    #     weights_save_path=None,
    #     max_epochs=args.num_epochs,
    #     resume_from_checkpoint="./lightning_logs/version_74/checkpoints/epoch=63-ade_val=1.66-fde_val=3.53.ckpt"
    # )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
