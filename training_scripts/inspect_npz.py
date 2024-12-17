import torch
import numpy as np
from collections import OrderedDict
import timm

def inspect_npz(file_path):
    # Load the .npz file
    data = np.load(file_path)
    
    print(f"Contents of {file_path}:")
    print("------------------------")
    
    # Iterate through all arrays in the file
    for key in data.files:
        array = data[key]
        print(f"Array name: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        
        # Print first few elements if array is not empty
        if array.size > 0:
            print("  First few elements:")
            print(array.flatten()[:5])  # Show first 5 elements
        else:
            print("  (Empty array)")
        
        print()  # Add a blank line for readability

    data.close()  # Close the file

def load_npz_to_pytorch(npz_path, model_name='vit_base_patch16_224'):
    # Load the .npz file
    npz_data = np.load(npz_path)
    
    # Create a PyTorch model
    model = timm.create_model(model_name, pretrained=False)
    
    # Create a new state dict for PyTorch
    pytorch_state_dict = OrderedDict()
    
    # Mapping between .npz keys and PyTorch keys
    key_mapping = {
        'embedding/kernel': 'patch_embed.proj.weight',
        'embedding/bias': 'patch_embed.proj.bias',
        'cls': 'cls_token',
        'Transformer/encoder_norm/scale': 'norm.weight',
        'Transformer/encoder_norm/bias': 'norm.bias',
        'head/kernel': 'head.weight',
        'head/bias': 'head.bias',
        'pre_logits/kernel': 'pre_logits.fc.weight',
        'pre_logits/bias': 'pre_logits.fc.bias',
    }
    
    # Helper function to reshape weights if necessary
    def reshape_conv_weights(weights):
        return weights.transpose(3, 2, 0, 1)
    
    for npz_key, array in npz_data.items():
        if npz_key in key_mapping:
            pytorch_key = key_mapping[npz_key]
            if 'kernel' in npz_key and len(array.shape) == 4:
                array = reshape_conv_weights(array)
            elif npz_key == 'head/kernel':
                array = array.T  # Transpose for linear layer
            pytorch_state_dict[pytorch_key] = torch.from_numpy(array)
        elif npz_key.startswith('Transformer/encoderblock_'):
            parts = npz_key.split('/')
            block_num = int(parts[1].split('_')[1])
            layer_name = parts[2]
            weight_name = parts[3]
            
            if layer_name == 'MlpBlock_3':
                layer_name = 'mlp'
                if weight_name == 'kernel':
                    weight_name = 'fc1.weight' if 'Dense_0' in npz_key else 'fc2.weight'
                elif weight_name == 'bias':
                    weight_name = 'fc1.bias' if 'Dense_0' in npz_key else 'fc2.bias'
            elif layer_name == 'LayerNorm_0':
                layer_name = 'norm1'
            elif layer_name == 'LayerNorm_2':
                layer_name = 'norm2'
            elif layer_name == 'MultiHeadDotProductAttention_1':
                layer_name = 'attn'
                if 'key' in weight_name:
                    weight_name = 'k_proj.' + weight_name
                elif 'query' in weight_name:
                    weight_name = 'q_proj.' + weight_name
                elif 'value' in weight_name:
                    weight_name = 'v_proj.' + weight_name
                elif 'out' in weight_name:
                    weight_name = 'proj.' + weight_name
            
            pytorch_key = f'blocks.{block_num}.{layer_name}.{weight_name}'
            if 'kernel' in weight_name:
                array = array.T
            pytorch_state_dict[pytorch_key] = torch.from_numpy(array)
        else:
            print(f"Unhandled key: {npz_key}")
    
    # Load the weights into the model
    model.load_state_dict(pytorch_state_dict, strict=False)
    
    return model

# Usage
file_path = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz'
inspect_npz(file_path)

# model = load_npz_to_pytorch(file_path)


# Now you can use the model for inference or further training
# print(model)