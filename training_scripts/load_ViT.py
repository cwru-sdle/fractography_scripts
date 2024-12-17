import torch
import numpy as np
import sys
import os

from collections import OrderedDict
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import ResNet
from timm.models.layers import PatchEmbed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def load_npz_to_pytorch(npz_path):
    npz_data = np.load(npz_path)
    model = HybridResNetViT()
    pytorch_state_dict = OrderedDict()

    for npz_key, array in npz_data.items():
        if npz_key.startswith('block'):
            parts = npz_key.split('/')
            block_num = int(parts[0][5:])
            unit_num = int(parts[1][4:])
            layer_name = parts[2]
            weight_type = parts[3]

            if layer_name.startswith('conv'):
                pytorch_key = f'resnet.layer{block_num}.{unit_num-1}.{layer_name}.weight'
                array = array.transpose(3, 2, 0, 1)
            elif layer_name.startswith('gn'):
                if weight_type == 'scale':
                    pytorch_key = f'resnet.layer{block_num}.{unit_num-1}.{layer_name}.weight'
                else:
                    pytorch_key = f'resnet.layer{block_num}.{unit_num-1}.{layer_name}.bias'
            else:
                continue

        elif npz_key.startswith('Transformer'):
            parts = npz_key.split('/')
            if parts[1] == 'posembed_input':
                pytorch_key = 'vit.pos_embed'
            elif parts[1] == 'encoder_norm':
                pytorch_key = f'vit.norm.{"weight" if parts[2] == "scale" else "bias"}'
            elif parts[1].startswith('encoderblock_'):
                block_num = int(parts[1].split('_')[1])
                layer_name = parts[2]
                weight_type = parts[3]

                if layer_name == 'MultiHeadDotProductAttention_1':
                    if 'query' in weight_type:
                        pytorch_key = f'vit.blocks.{block_num}.attn.q_proj.{"weight" if "kernel" in weight_type else "bias"}'
                    elif 'key' in weight_type:
                        pytorch_key = f'vit.blocks.{block_num}.attn.k_proj.{"weight" if "kernel" in weight_type else "bias"}'
                    elif 'value' in weight_type:
                        pytorch_key = f'vit.blocks.{block_num}.attn.v_proj.{"weight" if "kernel" in weight_type else "bias"}'
                    elif 'out' in weight_type:
                        pytorch_key = f'vit.blocks.{block_num}.attn.proj.{"weight" if "kernel" in weight_type else "bias"}'
                elif layer_name.startswith('MlpBlock_3'):
                    if 'Dense_0' in npz_key:
                        pytorch_key = f'vit.blocks.{block_num}.mlp.fc1.{"weight" if "kernel" in weight_type else "bias"}'
                    else:
                        pytorch_key = f'vit.blocks.{block_num}.mlp.fc2.{"weight" if "kernel" in weight_type else "bias"}'
                elif layer_name.startswith('LayerNorm_0'):
                    pytorch_key = f'vit.blocks.{block_num}.norm1.{"weight" if weight_type == "scale" else "bias"}'
                elif layer_name.startswith('LayerNorm_2'):
                    pytorch_key = f'vit.blocks.{block_num}.norm2.{"weight" if weight_type == "scale" else "bias"}'
            else:
                continue

        elif npz_key == 'cls':
            pytorch_key = 'vit.cls_token'
        elif npz_key.startswith('head'):
            pytorch_key = f'vit.head.{"weight" if "kernel" in npz_key else "bias"}'
        elif npz_key.startswith('pre_logits'):
            pytorch_key = f'vit.pre_logits.fc.{"weight" if "kernel" in npz_key else "bias"}'
        elif npz_key == 'conv_root/kernel':
            pytorch_key = 'resnet.conv1.weight'
            array = array.transpose(3, 2, 0, 1)
        elif npz_key.startswith('gn_root'):
            pytorch_key = f'resnet.bn1.{"weight" if "scale" in npz_key else "bias"}'
        elif npz_key.startswith('embedding'):
            pytorch_key = f'embedding.proj.{"weight" if "kernel" in npz_key else "bias"}'
            if 'kernel' in npz_key:
                array = array.transpose(3, 2, 0, 1)
        else:
            print(f"Unhandled key: {npz_key}")
            continue

        if 'kernel' in npz_key and 'conv' not in npz_key and len(array.shape) == 2:
            array = array.T

        if pytorch_key == 'resnet.bn1.weight' or pytorch_key == 'resnet.bn1.bias':
            array = array.squeeze()

        pytorch_state_dict[pytorch_key] = torch.from_numpy(array)

    # Adjust pos_embed if necessary
    if 'vit.pos_embed' in pytorch_state_dict:
        pos_embed = pytorch_state_dict['vit.pos_embed']
        if pos_embed.shape[1] != model.vit.pos_embed.shape[1]:
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.permute(0, 2, 1).unsqueeze(0), 
                size=(model.vit.pos_embed.shape[1],), 
                mode='linear'
            ).squeeze(0).permute(0, 2, 1)
            pytorch_state_dict['vit.pos_embed'] = pos_embed

    model.load_state_dict(pytorch_state_dict, strict=False)
    return model
# Usage
npz_path = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz'
# model = load_npz_to_pytorch(npz_path)
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
model = HybridResNetViT()
model.load_state_dict(torch.load(npz_path))
print(model)