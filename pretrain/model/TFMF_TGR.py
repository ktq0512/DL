
import numpy as np
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import conv
import pytorch_lightning as pl

from torch_geometric.utils import from_scipy_sparse_matrix

from scipy import sparse
import math


# Get the paths of the repository
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))


class TMFModel(pl.LightningModule):
    def __init__(self, args):
        super(TMFModel, self).__init__() # allows us to avoid using the base class name explicitly.
        self.args = args
        self.cut = self.args.num_restore - self.args.num_preds
        self.save_hyperparameters() # It will enable Lightning to store all the provided arguments under the self.hparams attribute. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        self.linear_embedding = LinearEmbedding(3,self.args)
        self.pos_encoder= PositionalEncoding1D(self.args.latent_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.args.latent_size))
        self.mask_ratio = 0.75
        self.encoder_transformer = EncoderTransformer(self.args)
        # self.out_concat=Concattensor(self.args)
        self.spanet = EncoderTransformer(self.args)

        self.cross_attender = CrossAttender(self.args.latent_size,sum(self.args.mod_steps))

        # self.cross_generate = Cross_generate(self.args)
        # self.decoder_tranformer = Decoder_Transformer(self.args)
        self.mlp = MLP(self.args)
        self.mlp_mask = MLP_Mask(self.args)
       
        self.agent_gnn = AgentGnn(self.args)
     

        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.class_loss = nn.CrossEntropyLoss()
        self.is_frozen = False
         

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test"))
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test_pre_clean.pkl"))
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=72)
        parser_training.add_argument(
            "--lr_values", type=list, default=[2e-4])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[])
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=16)
        parser_training.add_argument("--val_batch_size", type=int, default=16)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--latent_size", type=int, default=128)
        parser_model.add_argument("--num_preds", type=int, default=25)
        parser_model.add_argument("--num_restore", type=int, default=49)

        parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=180)

        return parent_parser

    def forward(self, batch):
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen
        if self.is_frozen:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()

        # displ：List，并且length=batch_size，List内每个元素都是tensor，shape为（agent_num,T,feature_num=3）
        displ = batch["displ"]
        displ_for_mask = [x[0] for x in displ]

        
        agents_per_sample = [x.shape[0] for x in displ]
        displ_cat = torch.cat(displ,dim=0)
        displ_cat = displ_cat[:,:self.cut,:]
        displ_for_mask_cat = torch.stack(displ_for_mask,dim=0)

        linear_output = self.linear_embedding(displ_cat)
        pos_encodeing = self.pos_encoder(linear_output)
        pos_encodeing = pos_encodeing + linear_output

        # tensor ，out_transformer shape：（sum（agents_per_sample），latent_size）
        out_transformer = self.encoder_transformer(pos_encodeing)
        out_transformer = out_transformer[:,-1,:]
        out_transformer = torch.split(out_transformer, agents_per_sample)
        out_transformer = list(out_transformer)
        
        max_agent_nums = max(agents_per_sample)
        padded_transformer = []
        for i in range(len(agents_per_sample)):
            padding_size = max_agent_nums - out_transformer[i].shape[0]
            pad = (0, 0, 0, padding_size)
            pad_transformer = F.pad(out_transformer[i], pad, mode='constant', value=0)
            padded_transformer.append(pad_transformer)
        input_spanet = torch.stack(padded_transformer,dim=0)     
        out_spanet = self.spanet(input_spanet)  

        mlp_input = out_spanet.view(-1,out_spanet.shape[-1])
        out_mlp = self.mlp(mlp_input)
        out_mlp = out_mlp.view(len(displ), -1, self.args.num_preds, 2)
        out = []
        for i in range(len(displ)):
            nums = agents_per_sample[i]
            out.append(out_mlp[i][:nums,:,:])

        # 先mask后入encoder
        linear_output_for_mask = self.linear_embedding(displ_for_mask_cat)
        linear_mask, mask = random_mask(linear_output_for_mask,self.mask_token,self.mask_ratio)   
        pos_encodeing_for_mask = self.pos_encoder(linear_mask)
        out_transformer_for_mask = self.encoder_transformer(pos_encodeing_for_mask)
        
        mlp_for_mask_input = out_transformer_for_mask.view(-1,out_transformer_for_mask.shape[-1])
        out_mlp_for_mask = self.mlp_mask(mlp_for_mask_input)
        
        out_for_mask = out_mlp_for_mask.view(len(displ),self.args.num_restore, 2) 
       
        
        return out,out_for_mask,mask

    def freeze(self):
        # for param in self.parameters():
        #     param.requires_grad = False

        # self.decoder_residual.unfreeze_layers()
        for name, param in self.named_parameters():
            if name.startswith('mlp') or name.startswith('cross_attender'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.is_frozen = True

    def prediction_loss(self, preds, gts, redispl_masked, displ_cat_masked):
        preds_cat = torch.cat(preds,dim=0)
        gt_cat = torch.cat(gts,dim=0)
        loss_out = self.reg_loss(preds_cat, gt_cat)
        loss_out = torch.sum(torch.sum(loss_out, dim=2), dim=1)
        loss_out = torch.mean(loss_out)

        # # Stack all the predicted trajectories of the target agent
        # num_mods = preds.shape[2]
        # # [0] is required to remove the unneeded dimensions
        # preds = torch.cat([x[0] for x in preds], 0)
        # # Stack all the true trajectories of the target agent
        # # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # # to the target agent
        # gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        # gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0) # repeate the gt for all ks 
        # loss_single = self.reg_loss(preds, gt_target)
        # loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)
        # loss_single = torch.split(loss_single, num_mods)
        # # Tuple to tensor
        # loss_single = torch.stack(list(loss_single), dim=0)
        # min_loss_index = torch.argmin(loss_single, dim=1)
        # min_loss_combined = [x[min_loss_index[i]]
        #                      for i, x in enumerate(loss_single)]
        # # loss_out = torch.sum(torch.stack(min_loss_combined))
        # loss_out = torch.sum(torch.stack(min_loss_combined))/len(min_loss_combined)

        # loss for mask
        loss_mask = self.reg_loss(redispl_masked, displ_cat_masked)
        loss_mask = torch.sum(torch.sum(loss_mask, dim=2), dim=1)
        loss_mask = loss_mask.mean()
       
        
        loss = loss_out + loss_mask
        return loss

    def configure_optimizers(self):

        
        if self.current_epoch == self.args.mod_freeze_epoch:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), weight_decay=self.args.wd)
        return optimizer

    def on_train_epoch_start(self):
        print(self.current_epoch)
        # Trigger weight freeze and optimizer reinit on mod_freeze_epoch
        if self.current_epoch == self.args.mod_freeze_epoch:
            self.freeze()
            self.trainer.accelerator.setup_optimizers(self.trainer)


        # Set learning rate according to current epoch
        for single_param in self.optimizers().param_groups:
            single_param["lr"] = self.get_lr(self.current_epoch)


    def training_step(self, train_batch, batch_idx):
        out,out_for_mask,mask = self.forward(train_batch)
        gts = []
        for i in range(len(train_batch["displ"])):
            gts.append(train_batch["displ"][i][:,self.cut:self.args.num_restore,:2])


        displ = train_batch["displ"]
        displ = [x[0] for x in displ]
        displ_cat = torch.stack(displ, dim=0)
        displ_cat = displ_cat[:, :, :2]
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, 2)
        # print(out.shape,displ_cat.shape,mask.shape)
        redispl_masked = out_for_mask * mask
        displ_cat_masked = displ_cat * mask

        loss = self.prediction_loss(out, gts, redispl_masked, displ_cat_masked)
        # self.log("loss_train", loss / len(out))
        self.log("loss_train", loss)
        # for name, param in self.named_parameters():
        #     print("name:"+str(name)+"   "+str(param.requires_grad))
            
        return loss
    
    def training_epoch_end(self, training_step_outputs):
        # 计算整个 epoch 的平均损失
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()

        # 使用 self.log 记录整个 epoch 的平均损失
        self.log('loss_train_epoch', avg_loss, prog_bar=True)

    def get_lr(self, epoch):
        lr_index = 0
        for lr_epoch in self.args.lr_step_epochs:
            if epoch < lr_epoch:
                break
            lr_index += 1
        return self.args.lr_values[lr_index]

    def validation_step(self, val_batch, batch_idx):
        out,out_for_mask,mask = self.forward(val_batch)
        
        gts = []
        for i in range(len(val_batch["displ"])):
            gts.append(val_batch["displ"][i][:,self.cut:self.args.num_restore,:2])


        displ = val_batch["displ"]
        displ = [x[0] for x in displ]
        displ_cat = torch.stack(displ, dim=0)
        displ_cat = displ_cat[:, :, :2]
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, 2)
        # print(out.shape,displ_cat.shape,mask.shape)
        redispl_masked = out_for_mask * mask
        displ_cat_masked = displ_cat * mask


        loss = self.prediction_loss(out, gts, redispl_masked, displ_cat_masked)
        # self.log("loss_val", loss / len(out))
        self.log("loss_val", loss, prog_bar=False)
        # Extract target agent only
        pred = []
        gt = []
        for i in range(len(out)):
            pred.append(out[i][0])
            gt.append(gts[i][0])
        pred = torch.stack(pred,dim=0)
        gt = torch.stack(gt,dim=0)
        
        
        return pred, gt, loss

    def validation_epoch_end(self, validation_outputs):
        
        pred = [out[0] for out in validation_outputs]
        pred = torch.cat(pred,dim=0)
        gt = [out[1] for out in validation_outputs]
        gt = torch.cat(gt,dim=0)
        batch_loss = [out[2] for out in validation_outputs]
        epoch_loss = torch.stack(batch_loss).mean()

        ade_norm = torch.norm(pred - gt, dim=2)
        ade = torch.mean(ade_norm)
        fde_norm = ade_norm[:,-1]
        fde = torch.mean(fde_norm)


  
        self.log("ade_val", ade, prog_bar=True)
        self.log("fde_val", fde, prog_bar=True)
        self.log("loss_val_epoch", epoch_loss, prog_bar=True)

    



class LinearEmbedding(nn.Module):
    def __init__(self,input_size,args):
        super(LinearEmbedding, self).__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = args.latent_size

        self.encoder_input_layer = nn.Linear(
                in_features=self.input_size, 
                out_features=self.output_size 
                    )
    def forward(self,linear_input):

        linear_out = F.relu(self.encoder_input_layer(linear_input))

        return linear_out 


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)

        return self.cached_penc

class EncoderTransformer(nn.Module):
    def __init__(self, args):
        super(EncoderTransformer, self).__init__()
        self.args = args

        self.d_model = self.args.latent_size # embedding dimension
        self.nhead = 128
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 1
        self.dropout = 0.1

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, transformer_in):

        transformer_out = F.relu(self.transformer_encoder(transformer_in))
        return transformer_out

class CrossAttender(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, dim)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        q = self.query_embed.weight.unsqueeze(0)
        kv = self.kv(x)
        k, v = kv.split(self.dim, dim=-1)
        q = self.q_proj(q)

        attn_weights = (q @ k.transpose(-2, -1)) / self.dim**0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        x = (attn_weights @ v).transpose(1, 2).reshape(-1, self.num_queries, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def random_mask(x, mask_token, mask_ratio):
    N, L, D = x.shape  # batch, length, dim
    mask = torch.rand((N, L), device=x.device) < mask_ratio
    # Apply the mask to the input
    x_masked = x.clone()
    x_masked[mask] = mask_token

    return x_masked, mask

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.BatchNorm1d(self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.BatchNorm1d(self.latent_size) # Batch normalization solves a major problem called internal covariate shift. 

        self.output_fc = nn.Linear(self.latent_size, args.num_preds * 2)
        

    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        prednet_out = self.output_fc(x)

        return prednet_out # MLP_out: 16,6,120

class MLP_Mask(nn.Module):
    def __init__(self, args):
        super(MLP_Mask, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.BatchNorm1d(self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.BatchNorm1d(self.latent_size) # Batch normalization solves a major problem called internal covariate shift. 

        self.output_fc = nn.Linear(self.latent_size, 2)
        

    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        prednet_out = self.output_fc(x)

        return prednet_out # MLP_out: 16,6,120


class AgentGnn(nn.Module):
    def __init__(self, args):
        super(AgentGnn, self).__init__()
        self.args = args
        self.latent_size = args.latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

    def forward(self, gnn_in, centers, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
            agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        edge_index_out1 = []
        for i in agents_per_sample:
            edge_index_out1.append(gnn_out[0:i,:])
            gnn_out = gnn_out[i:,:]

        return edge_index_out1

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):

            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        

        return edge_index
    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = data[cols] - data[rows]

        return edge_attr