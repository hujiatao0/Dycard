import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = 2 ** math.ceil(math.log2(embed_dim * 2))
        
        # Q and K share parameters
        self.qk_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.mask_mlp = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        self.scale = math.sqrt(self.hidden_dim)
    
    def forward(self, x, padding_mask=None, attention_mask=None, seq_lengths=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            padding_mask: Original padding mask (batch_size, 1, seq_len)
            attention_mask: Custom attention mask (batch_size, seq_len, seq_len)
            seq_lengths: List of valid sequence lengths for each sample
        """
        # Project to Q, K, V spaces
        qk = self.qk_proj(x)  # Q and K share parameters
        v = self.v_proj(x)
        
        # Calculate attention scores
        attn_weights = torch.matmul(qk, qk.transpose(-2, -1)) / self.scale
        # print(f"attn_weights shape: {attn_weights.shape}")
        # print(f"attn_weights: {attn_weights}")

        # if attention_mask is not None and seq_lengths is not None:
        #     batch_size = attn_weights.size(0)
        #     for i in range(batch_size):
        #         valid_len = seq_lengths[i]
        #         attn_weights[i] = attn_weights[i] * attention_mask[i]
        
        # Apply padding mask (if provided)
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
        # print(f" before softmax attn_weights: {attn_weights}")
            
        # Scale attention mask to appropriate range before applying
        # if attention_mask is not None and seq_lengths is not None:
        #     batch_size = attn_weights.size(0)
        #     for i in range(batch_size):
        #         # print(f"attention mask: {attention_mask[i]}")
        #         # print(f"attention weights: {attn_weights[i]}")
        #         valid_len = seq_lengths[i]
        #         # Only calculate std when valid length > 1
        #         if valid_len > 1:  # Ensure at least two samples
        #             valid_weights = attn_weights[i, :valid_len, :valid_len]
        #             # Use unbiased estimation for std
        #             std = valid_weights.std(unbiased=True).item()
        #             if std != 0 and not math.isnan(std):
        #                 scaled_mask = attention_mask[i] * std
        #                 attn_weights[i] = attn_weights[i] + scaled_mask
        #         else:
        #             # Use default scale factor when sequence is too short
        #             std = 1.0
        #             scaled_mask = attention_mask[i] * std
        #             attn_weights[i] = attn_weights[i] + scaled_mask
                # print(f"scaled_mask: {scaled_mask}")

                # print(f"attn_weights: {attn_weights[i]}")

        if attention_mask is not None and seq_lengths is not None:
            batch_size = attn_weights.size(0)
            for i in range(batch_size):
                projected_mask = self.mask_mlp(attention_mask[i].unsqueeze(-1)).squeeze(-1)
                valid_len = seq_lengths[i]
                if valid_len > 1:
                    valid_weights = attn_weights[i, :valid_len, :valid_len]
                    std = valid_weights.std(unbiased=True).item()
                    if std != 0 and not math.isnan(std):
                        scaled_mask = projected_mask * std
                        attn_weights[i] = attn_weights[i] + scaled_mask
                else:
                    scaled_mask = projected_mask * 1.0
                    attn_weights[i] = attn_weights[i] + scaled_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        # print(f" after softmax attn_weights: {attn_weights}")
        # Apply attention weights
        output = torch.matmul(attn_weights, v)  
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output, attn_weights

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleAttention(embed_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, padding_mask=None, attention_mask=None, seq_lengths=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            padding_mask: Padding mask (batch_size, 1, seq_len)
            attention_mask: Custom attention mask (batch_size, seq_len, seq_len)
            seq_lengths: List of valid sequence lengths for each sample
        """
        attentions = []
        # print(f"x shape: {x.shape}")
        # print(f"x: {x}")
        # print(f"padding_mask shape: {padding_mask.shape}")
        # print(f"padding_mask: {padding_mask}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        # print(f"attention_mask: {attention_mask}")
        # print(f"seq_lengths: {seq_lengths}")
        for layer in self.layers:
            # Residual connection
            residual = x
            x, attn = layer(x, padding_mask, attention_mask, seq_lengths)
            x = residual + x
            x = self.norm(x)
            attentions.append(attn)
            
        return x, attentions
    
class PredictionModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim=16, scale_factor=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),  # +1 for log-transformed initial estimate
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights close to 0 and biases to 0
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, features, initial_estimate):
        # features: [batch_size, embed_dim]
        # initial_estimate: [batch_size, 1]
        
        # Log transform initial estimate
        log_estimate = torch.log1p(initial_estimate)
        scaled_estimate = log_estimate * self.scale_factor
        
        # Concatenate features and log-transformed initial estimate
        combined = torch.cat([features, scaled_estimate], dim=1)
        # print(f"combined: {combined}")
        
        # Predict true value in log space
        log_delta = self.fusion(combined)
        
        # Convert delta back to original space
        delta = torch.expm1(log_estimate + log_delta) - torch.expm1(log_estimate)
        
        # Final prediction = initial estimate + delta
        prediction = initial_estimate + delta
        prediction = torch.clamp(prediction, min=1.0)
        
        return prediction, delta
    
class CompleteModel(nn.Module):
    def __init__(self, embed_dim, num_layers=1, hidden_dim=16):
        super().__init__()
        self.attention_model = AttentionModel(embed_dim, num_layers)
        self.feature_mapping = nn.Linear(embed_dim, embed_dim)
        self.prediction_model = PredictionModel(embed_dim, hidden_dim)
        
    def forward(self, x, initial_estimate, padding_mask=None, attention_mask=None, seq_lengths=None):
        """
        Args:
            x: Input sequence [batch_size, seq_len, embed_dim]
            initial_estimate: Initial estimate value [batch_size, 1]
            padding_mask: Padding mask [batch_size, 1, seq_len]
            attention_mask: Attention mask
            seq_lengths: List of valid sequence lengths for each sample
        """
        # Get attention features
        # features shape: [batch_size, seq_len, embed_dim]
        # print(f"x shape: {x.shape}")
        # print(f"x: {x}")
        features, attentions = self.attention_model(x, padding_mask, attention_mask, seq_lengths)  
        
        # Convert sequence features to single feature vector
        if padding_mask is not None:
            # padding_mask is [batch_size, 1, seq_len] boolean tensor, True means padding
            # Convert to [batch_size, seq_len, 1] float tensor, where non-padding positions are 1
            mask = (~padding_mask).squeeze(1).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            # Normalize weights
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-9)
            # Weighted average
            features = (features * mask).sum(dim=1)  # [batch_size, embed_dim]
        else:
            # If no padding mask, use average pooling
            features = features.mean(dim=1)  # [batch_size, embed_dim]
        
        # Feature mapping through fully connected layer
        features = self.feature_mapping(features)
        
        # Prediction
        prediction, delta = self.prediction_model(features, initial_estimate)
        
        return prediction, delta, attentions
    
class CombinedLoss(nn.Module):
    def __init__(self, delta_weight=1.5):
        super().__init__()
        self.delta_weight = delta_weight
        
    def forward(self, prediction, target, delta):
        # Calculate error in log space
        # print(f"target : {target}")
        log_pred = torch.log10(prediction + 1)
        log_target = torch.log10(target + 1)
        scale = self.delta_weight ** log_target
        base_loss = (log_pred - log_target) ** 2
        # print(f"base_loss: {base_loss}")
        # print(f"scale: {scale}")

        total_loss = torch.mean(base_loss * scale) 
        # print(f"total_loss: {total_loss}")
        return total_loss
    
def create_padding_mask(input_ids, padding_idx=0):
    """
    Create padding mask from input_ids
    Args:
        input_ids: Input sequence [batch_size, seq_len]
        padding_idx: Token id for padding, default 0
    Returns:
        attention_mask: [batch_size, 1, seq_len] 
    """
    # Create key padding mask [batch_size, seq_len]
    # True at padding_idx positions, indicating need to mask
    key_padding_mask = (input_ids == padding_idx)
    
    # Expand dims for broadcasting [batch_size, 1, seq_len]
    attention_mask = key_padding_mask.unsqueeze(1)
    
    return attention_mask

# Usage example
def example_usage():
    # Parameter settings
    embed_dim = 4
    num_epochs = 2
    
    # Create model and optimizer
    model = CompleteModel(embed_dim)
    criterion = CombinedLoss(delta_weight=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create embedding layers
    emb1 = nn.Embedding(10, embed_dim//2, padding_idx=0)
    emb2 = nn.Embedding(10, embed_dim//2, padding_idx=0)
    
    # Simulate multiple training rounds
    for epoch in range(num_epochs):
        # Simulate multiple batches
        for batch in range(3):
            # Construct input sequence with padding
            input_ids = torch.tensor([
                [1, 2, 3, 4, 5, 0, 0, 0],  # length 5
                [6, 7, 8, 0, 0, 0, 0, 0],  # length 3
                [1, 3, 5, 7, 9, 2, 0, 0],  # length 6
                [2, 4, 6, 8, 0, 0, 0, 0]   # length 4
            ])
            seq_lengths = [5, 3, 6, 4]  # Valid length for each sample
            batch_size, max_seq_len = input_ids.size()
            
            # Get features from embeddings
            vec1 = emb1(input_ids)
            vec2 = emb2(input_ids)
            x = torch.cat([vec1, vec2], dim=-1)
            
            # Create padding mask [batch_size, 1, seq_len]
            padding_mask = create_padding_mask(input_ids)
            
            # Create attention mask (optional additional attention constraints)
            attention_mask = torch.rand(batch_size, max_seq_len, max_seq_len) * 2 - 1
            
            # Simulate estimated and true values
            initial_estimate = torch.randint(4000, 6000, (batch_size, 1)).float()
            target = initial_estimate + torch.abs(torch.randn(batch_size, 1) * 5000)
            
            # Train one step
            model.train()
            optimizer.zero_grad()
            # print(f"x shape: {x.shape}")
            # print(f"initial_estimate: {initial_estimate}")
            # print(f"padding_mask shape: {padding_mask.shape}")
            # print(f"attention_mask shape: {attention_mask.shape}")
            # print(f"seq_lengths shape: {seq_lengths}")
            prediction, delta, attentions = model(x, initial_estimate, padding_mask, 
                                                attention_mask, seq_lengths)
            
            
            loss = criterion(prediction, target, delta)
            loss.backward()
            optimizer.step()
            
            # Print training info
            print(f"\nEpoch {epoch+1}, Batch {batch+1}:")
            print("Training examples:")
            print(f"Initial estimate: {initial_estimate.squeeze().tolist()}")
            print(f"Prediction: {prediction.squeeze().tolist()}")
            print(f"True value: {target.squeeze().tolist()}")
            print(f"Delta: {delta.squeeze().tolist()}")
            print(f"Loss: {loss.item():.4f}")
            
            # Evaluation metrics
            mae = torch.abs(prediction - target).mean().item()
            initial_mae = torch.abs(initial_estimate - target).mean().item()
            improvement = (initial_mae - mae) / initial_mae * 100
            
            print(f"Mean absolute error: {mae:.4f}")
            print(f"Relative improvement: {improvement:.2f}%")

def inference_test():
    # Initialize model
    embed_dim = 32
    model = CompleteModel(embed_dim, num_layers=10)
    model.eval()  # Set to evaluation mode
    
    # Prepare single sample data
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]])  # Sequence length 6, followed by padding
    seq_lengths = [6]
    # Create embedding layers
    emb1 = nn.Embedding(10, embed_dim//2, padding_idx=0)
    emb2 = nn.Embedding(10, embed_dim//2, padding_idx=0)
    
    # Get features
    vec1 = emb1(input_ids)
    vec2 = emb2(input_ids)
    x = torch.cat([vec1, vec2], dim=-1)
    
    # Prepare other inputs
    initial_estimate = torch.tensor([[50.0]])
    padding_mask = create_padding_mask(input_ids)  # Use new padding mask creation function
    
    # Create attention mask, keeping consistent with sequence length
    batch_size, seq_len = input_ids.size()
    attention_mask = torch.rand(batch_size, seq_len, seq_len)
    # Warmup once
    # with torch.no_grad():
    #     model(x, initial_estimate, padding_mask, attention_mask, seq_lengths)
    
    # Test inference time
    start_time = time.time()
    with torch.no_grad():
        prediction, delta, attentions = model(x, initial_estimate, padding_mask, attention_mask, seq_lengths)

    print(f"len attentions: {len(attentions)}")
    print(f"attentions: {attentions[0].shape}")
    end_time = time.time()
    
    print(f"Single inference time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Initial estimate: {initial_estimate.item():.2f}")
    print(f"Prediction: {prediction.item():.2f}")
    print(f"Delta: {delta.item():.2f}")


if __name__ == "__main__":
    # example_usage()
    # print("\n" + "="*50 + "\n")
    inference_test()
