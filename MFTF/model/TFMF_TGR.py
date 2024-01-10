
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

        self.save_hyperparameters() # It will enable Lightning to store all the provided arguments under the self.hparams attribute. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        self.linear_embedding = LinearEmbedding(3,self.args)
        self.pos_encoder= PositionalEncoding1D(self.args.latent_size)
        self.encoder_transformer = EncoderTransformer(self.args)
        # self.out_concat=Concattensor(self.args)
        # self.spanet_transformer = SPAnetTransformer(self.args)
        self.cross_attender = CrossAttender(self.args.latent_size,sum(self.args.mod_steps))

        # self.cross_generate = Cross_generate(self.args)
        # self.decoder_tranformer = Decoder_Transformer(self.args)
        self.mlp = MLP(self.args)
        self.mlp_score = MLP_Score(self.args)
        self.agent_gnn = AgentGnn(self.args)
        # self.decoder_residual = DecoderResidual(self.args)

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
        parser_training.add_argument("--num_epochs", type=int, default=70)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 1e-4,1e-5])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[40,50])
        parser_training.add_argument("--wd", type=float, default=0.01)
        # batch_size可调
        parser_training.add_argument("--batch_size", type=int, default=32)
        parser_training.add_argument("--val_batch_size", type=int, default=32)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        # 隐藏层维度，可调
        parser_model.add_argument("--latent_size", type=int, default=128)
        # encoder_transformer_layer_nums，可调
        parser_model.add_argument("--encoder_transformer_layer_nums", type=int, default=1)
        parser_model.add_argument("--num_preds", type=int, default=60)
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
        displ, centers = batch["displ"], batch["centers"]
        rotation, origin = batch["rotation"], batch["origin"]
        # print(displ[0].shape)
        # print(centers[0].shape)
        # Extract the number of agents in each sample of the current batch
        # List，并且length=batch_size，统计了displ中每个tensor的agent_num
        agents_per_sample = [x.shape[0] for x in displ]
        displ_cat = torch.cat(displ,dim=0)
        centers_cat = torch.cat(centers,dim=0)
        linear_output = self.linear_embedding(displ_cat)
        pos_encodeing = self.pos_encoder(linear_output)
        pos_encodeing = pos_encodeing + linear_output

        # tensor ，out_transformer shape：（sum（agents_per_sample），latent_size）
        out_transformer = self.encoder_transformer(pos_encodeing)
        out_transformer = out_transformer[:,-1,:]
        out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)

        max_agent_nums = max(agents_per_sample)
        padded_out_agent_gnn = []
       
        for i in range(len(out_agent_gnn)):
            padding_size = max_agent_nums - out_agent_gnn[i].shape[0]
            pad = (0, 0, 0, padding_size)
            pad_out_agent_gnn = F.pad(out_agent_gnn[i], pad, mode='constant', value=0)
            padded_out_agent_gnn.append(pad_out_agent_gnn)
        input_CA = torch.stack(padded_out_agent_gnn,dim=0)       

        # print(input_CA.shape)
        output_cross_attender = self.cross_attender(input_CA)
        # print(output_cross_attender.shape)

       
        mlp_input = output_cross_attender.view(-1,output_cross_attender.shape[-1])
        out_mlp = self.mlp(mlp_input)
        out_mlp_score = self.mlp_score(mlp_input)
        out = out_mlp.view(len(displ), 1, -1, self.args.num_preds, 2) #(16,1,6 num_mods,60 Hz, 2)
        out_scores = out_mlp_score.view(len(displ),  -1)
        out_scores = torch.nn.functional.softmax(out_scores, dim=1)
        # # List ，out_agent_gnn length：batch_size
        # out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)

        # tensor ，out_agent_gnn shape：（batch_size，latent_size）
        # out_agent_gnn= torch.stack([x[0] for x in out_agent_gnn])
        
        # tensor out_linear shape：（batch_size，1, 60x2）
        # out_linear = self.decoder_residual(out_agent_gnn, self.is_frozen)

        # out = out_linear.view(len(displ), 1, -1, self.args.num_preds, 2)

        # Iterate over each batch and transform predictions into the global coordinate frame, Matrix product of two tensors.
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].view(
                1, 1, 1, -1
            )
        return out, out_scores

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

    def prediction_loss(self, preds, gts, scores):
        # Stack all the predicted trajectories of the target agent
        num_mods = preds.shape[2]
        # [0] is required to remove the unneeded dimensions
        preds = torch.cat([x[0] for x in preds], 0)

        # Stack all the true trajectories of the target agent
        # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # to the target agent
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0) # repeate the gt for all ks 

        loss_single = self.reg_loss(preds, gt_target)
        loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

        loss_single = torch.split(loss_single, num_mods)

        # Tuple to tensor
        loss_single = torch.stack(list(loss_single), dim=0)

        min_loss_index = torch.argmin(loss_single, dim=1)
        
        # print(min_loss_index.shape)
        gt_scores = torch.zeros((min_loss_index.shape[0], 6))
        gt_scores = gt_scores.to(min_loss_index.device)
        # print(scores.shape)
        for i in range(min_loss_index.shape[0]):
            gt_scores[i][min_loss_index[i]] = 1
        
        # print(gt_scores.shape)
        min_loss_combined = [x[min_loss_index[i]]
                             for i, x in enumerate(loss_single)]
        # loss_out = torch.sum(torch.stack(min_loss_combined))
        loss_out = torch.sum(torch.stack(min_loss_combined))/len(min_loss_combined)
        loss_score = self.class_loss(scores,gt_scores)
        loss_out = loss_out+loss_score
        return loss_out


    def configure_optimizers(self):

        
        if self.current_epoch == self.args.mod_freeze_epoch:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), weight_decay=self.args.wd)

        # if self.current_epoch > 0:
        #     optimizer.load_state_dict(self.optimizer_state)

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
        out, scores = self.forward(train_batch)
        loss = self.prediction_loss(out, train_batch["gt"], scores)
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
        out, scores = self.forward(val_batch)
        loss = self.prediction_loss(out, val_batch["gt"],scores)
        # self.log("loss_val", loss / len(out))
        self.log("loss_val", loss, prog_bar=False)
        # Extract target agent only
        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        return pred, gt, loss

    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out[0] for out in validation_outputs]
        pred = np.concatenate(pred, 0)
        gt = [out[1] for out in validation_outputs]
        gt = np.concatenate(gt, 0)
        batch_loss = [out[2] for out in validation_outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt)
        # self.log("ade1_val", ade1, prog_bar=True)
        # self.log("fde1_val", fde1, prog_bar=True)
        self.log("ade_val", ade, prog_bar=True)
        self.log("fde_val", fde, prog_bar=True)
        self.log("loss_val_epoch", epoch_loss, prog_bar=True)

    def calc_prediction_metrics(self, preds, gts):
        # Calculate prediction error for each mode
        # Output has shape (batch_size, n_modes, n_timesteps)
        error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)

        # Calculate the error for the first mode (at index 0)
        fde_1 = np.average(error_per_t[:, 0, -1])
        ade_1 = np.average(error_per_t[:, 0, :])

        # Calculate the error for all modes
        # Best mode is always the one with the lowest final displacement
        lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
        error_per_t = error_per_t[np.arange(
            preds.shape[0]), lowest_final_error_indices]
        fde = np.average(error_per_t[:, -1])
        ade = np.average(error_per_t[:, :])

        return ade_1, fde_1, ade, fde



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
        self.num_layers = self.args.encoder_transformer_layer_nums
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



class Concattensor(nn.Module):
    def __init__(self,args):
        super(Concattensor, self).__init__()
        self.args=args
        self.latent_size=args.latent_size
    def forward(self, concat_in, agents_per_sample):
        tensor_list=[]
        for i in agents_per_sample:
            tensor_list.append(concat_in[0:i,:]) #拆分成16个场景，方便后面每个场景concat地图key-value，加完之后可以考虑转化为tensor方便后面计算
            concat_in=concat_in[i:,:]
        return tensor_list #(16,128)
    
class SPAnetTransformer(nn.Module):
    def __init__(self,args):
        super(SPAnetTransformer,self).__init__()
        self.args = args
        #self.enc_embedding=nn.Embedding(num_embeddings, embedding_dim)
        
        self.d_model = self.args.latent_size # embedding dimension
        self.nhead = 128
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 1
        self.dropout = 0.05

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, enc_in): #enc_in: tensor(16,128)
        enc_out = F.relu(self.transformer_encoder(enc_in))
        # enc_out1=enc_out[None,:,:].repeat(sum(self.args.mod_steps)) #6*16*128
        # enc_out1=torch.swapaxes(enc_out1, 0, 1) 
        return enc_out #enc_out: tensor(16,6,128)

class Cross_generate(nn.Module):
    def __init__(self, args):
        super(Cross_generate, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(args) #N=6，预测6条轨迹, 如果在这里不经过PredictionNet类，就生成N条轨迹预测，再输入Decoder?

        self.output = nn.ModuleList(output) # is just like a Python list. It was designed to store any desired number of nn.Module’s

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:

            sample_wise_out.append(self.output[0](decoder_in)) #training时，输入16*128入PredicionNet预测后6s轨迹
            

        decoder_out = torch.stack(sample_wise_out) #6*16*128
        decoder_out = torch.swapaxes(decoder_out, 0, 1) #swapaxes=transpose, 16*6*128

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True

class Decoder_Transformer(nn.Module):
    def __init__(self,args):
        super(Decoder_Transformer,self).__init__()
        self.args = args
        #self.enc_embedding=nn.Embedding(num_embeddings, embedding_dim)
        
        self.d_model = self.args.latent_size # embedding dimension
        self.nhead = 128
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 2
        self.dropout = 0.05

        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, enc_in, dec_in): #enc_in: tensor(16,6,128)
        dec_out = F.relu(self.transformer_decoder(enc_in, dec_in)) 
        return dec_out #enc_out: tensor(16,6,128)

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

class MLP_Score(nn.Module):
    def __init__(self, args):
        super(MLP_Score, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.BatchNorm1d(self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.BatchNorm1d(self.latent_size) 

        self.output_fc = nn.Linear(self.latent_size, 1)
        

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

class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output) # is just like a Python list. It was designed to store any desired number of nn.Module’s

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:

            sample_wise_out.append(self.output[0](decoder_in))
            

        decoder_out = torch.stack(sample_wise_out)
        decoder_out = torch.swapaxes(decoder_out, 0, 1)

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True


class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size) # Batch normalization solves a major problem called internal covariate shift. 

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

        return prednet_out
