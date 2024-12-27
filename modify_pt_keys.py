import torch

def remove_keys(state_dict, keys_to_remove):
    """
    Remove specified keys from the state dictionary.
    
    Args:
        state_dict (dict): The state dictionary to modify.
        keys_to_remove (list of str): List of keys to remove from the state dictionary.
        
    Returns:
        dict: The modified state dictionary with specified keys removed.
    """
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    return state_dict

# Load the checkpoint file
path = r'd:/Code/mobilenetv2_100_ra.pth'
checkpoint = torch.load(path)

# Specify the keys you want to remove
keys_to_remove = ["conv_head.weight", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked", "classifier.weight", "classifier.bias"]


# Remove the specified keys
# new_checkpoint = remove_keys(checkpoint, keys_to_remove)
for key in keys_to_remove:
    if key in checkpoint:
        del checkpoint[key]

# Save the new checkpoint file
torch.save(checkpoint, 'new_checkpoint.pt')

print("New checkpoint saved as 'new_checkpoint.pt'")
