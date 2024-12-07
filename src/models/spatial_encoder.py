import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spatial awareness"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SpatialAttention(nn.Module):
    """
    Multi-head attention module for spatial relationships
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        self.attention_weights = attn.detach()  # Store for visualization
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # Project to output dimension
        out = self.o_proj(out)
        
        # Add residual connection and layer normalization
        out = self.layer_norm(x + out)
        
        return out
    
    def get_attention_weights(self):
        """Return the last computed attention weights"""
        return self.attention_weights

class ObjectEncoder(nn.Module):
    """
    Encoder for object properties with spatial awareness
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.attention = SpatialAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Apply attention
        x = self.attention(x.unsqueeze(0)).squeeze(0)
        
        return x

class PhysicsGNN(nn.Module):
    """
    Graph Neural Network for physics-based refinement
    """
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2D distance vector
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # node_i + node_j + edge_ij
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Node update layers
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # node + message
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, node_dim)
        
    def forward(self, x):
        batch_size, num_nodes, node_dim = x.size()
        
        # Initialize node features
        node_features = self.node_encoder(x)
        
        # Compute pairwise distances for edge features
        pos = x[..., -2:]  # Last 2 dimensions are x,y coordinates
        dist = pos.unsqueeze(2) - pos.unsqueeze(1)  # (batch, num_nodes, num_nodes, 2)
        edge_features = self.edge_encoder(dist)
        
        # Message passing
        for layer in range(self.num_layers):
            # Compute messages
            node_i = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            node_j = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            
            messages = torch.cat([node_i, node_j, edge_features], dim=-1)
            messages = self.message_layers[layer](messages)
            
            # Aggregate messages (mean)
            messages = messages.mean(dim=2)
            
            # Update nodes
            node_features = self.update_layers[layer](
                torch.cat([node_features, messages], dim=-1)
            )
        
        # Project to output dimension
        output = self.output_layer(node_features)
        
        # Add residual connection for stability
        output = x + output
        
        return output