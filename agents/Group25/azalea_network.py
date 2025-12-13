import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block from Azalea"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AzaleaNetwork(nn.Module):
    """
    Azalea Hex network architecture.
    Input: (batch, 11, 11) long tensor with 0=empty, 1=player, 2=opponent
    Output: policy logits (batch, 121), value (batch, 1)
    """
    def __init__(self, board_size=11, num_blocks=6, num_channels=64):
        super().__init__()
        self.board_size = board_size
        
        # Input embedding: 3 channels (empty, player, opponent)
        self.input_conv = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 11, 11) long tensor with values 0, 1, 2
        Returns:
            policy: (batch, 121) logits
            value: (batch, 1) value in [-1, 1]
        """
        batch_size = x.shape[0]
        
        # Convert input to one-hot encoding
        # Create 3 channels: [empty, player, opponent]
        x_onehot = torch.zeros(batch_size, 3, self.board_size, self.board_size, 
                               dtype=torch.float32, device=x.device)
        x_onehot[:, 0, :, :] = (x == 0).float()  # Empty
        x_onehot[:, 1, :, :] = (x == 1).float()  # Player
        x_onehot[:, 2, :, :] = (x == 2).float()  # Opponent
        
        # Shared trunk
        x = F.relu(self.input_bn(self.input_conv(x_onehot)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(batch_size, -1)
        policy = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(batch_size, -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value


def load_azalea_model(model_path, device="cpu"):
    """
    Load pretrained Azalea model.
    
    Args:
        model_path: Path to .pth file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: AzaleaNetwork ready for inference
    """
    # Create model
    model = AzaleaNetwork(board_size=11, num_blocks=6, num_channels=64)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.to(device)
    model.eval()
    
    return model
