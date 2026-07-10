import torch
import torch.nn as nn


class CnnAttnDepthEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 32,
                 proprio_dim: int = 48, estimator_dim: int = 3, num_heads: int = 4):
        """
        Args:
            latent_dim: Output dimension of the encoder (default 32 in ActorCritic).
            hidden_dim: Internal embedding dimension for attention.
            patch_size: Size of patches to extract from depth map.
            proprio_dim: Dimension of the proprioceptive observation (current state).
            estimator_dim: Dimension of the estimator output (vel + latent).
            num_heads: Number of attention heads.
        """
        super().__init__()

        # Patch embedding
        # Input depth: [B, 1, 24, 32] -> [B, hidden_dim, H', W']
        self.cnn = nn.Sequential(
            # 16*24*1
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            nn.ELU(),
            # 8*12*8
            nn.Conv2d(8, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ELU(),
            # 8*12*32
        )

        # Positional encoding
        num_patches = 8 * 12
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Query projection (Proprio + Estimator -> Query Input)
        # We project the concatenated vector to hidden_dim to serve as the Query input for MHA
        print(f"AttnDepthEncoder input dims - Proprio: {proprio_dim}, Estimator: {estimator_dim}")
        self.query_input_proj = nn.Sequential(
            nn.Linear(proprio_dim + estimator_dim, 64),
            nn.ELU(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
        )

        # Multi-head Attention
        # embed_dim=hidden_dim, num_heads=num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ELU(),
        )

    def forward(self, depths, proprio, vel):
        # depths: [B, H, W, 1] -> [B, 1, H, W]
        # (Handle potential lack of channel dim if necessary, but HimDepthActorCritic passes expected shape?
        # Actually HimDepthActorCritic passes [B*stack, 1, H, W] via view, or [B*stack, H, W]?)
        # Let's check dims.
        depths = depths.permute(0, 3, 1, 2)

        # Patch embed -> [B, hidden_dim, H', W']
        x = self.cnn(depths)
        # Flatten spatial dims: [B, hidden_dim, N] where N = H'*W'
        x = x.flatten(2)
        # Permute to [B, N, C] for batch_first=True
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.pos_embed

        key_value = x

        # Create Query Input
        query_raw = torch.cat([proprio, vel], dim=-1)
        query = self.query_input_proj(query_raw)  # [B, hidden_dim]
        query = query.unsqueeze(1)  # [B, 1, hidden_dim]

        # Attention
        # query: [B, 1, hidden_dim]
        # key, value: [B, N, hidden_dim]
        # MHA returns (attn_output, attn_output_weights)
        attn_output, _ = self.mha(query, key_value, key_value, need_weights=False)

        # attn_output: [B, 1, hidden_dim]
        out = attn_output.squeeze(1)  # [B, hidden_dim]

        return self.out_proj(out)

    def get_attention_weights(self, depths, proprio, vel):
        """Get attention weights for visualization purposes.

        Args:
            depths: Depth map tensor [B, H, W, 1] or [B, 1, H, W]
            proprio: Proprioceptive observation [B, proprio_dim]
            vel: Velocity estimate [B, vel_dim]
            latent: Latent representation [B, latent_dim]

        Returns:
            attn_weights: Attention weights [B, 1, num_patches] or [B, num_heads, 1, num_patches] if num_heads > 1
        """
        # Process depths same as forward
        depths = depths.permute(0, 3, 1, 2)

        # Patch embed -> [B, hidden_dim, H', W']
        x = self.cnn(depths)
        # Flatten spatial dims: [B, hidden_dim, N] where N = H'*W'
        x = x.flatten(2)
        # Permute to [B, N, C] for batch_first=True
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.pos_embed

        key_value = x

        # Create Query Input
        query_raw = torch.cat([proprio, vel], dim=-1)
        query = self.query_input_proj(query_raw)  # [B, hidden_dim]
        query = query.unsqueeze(1)  # [B, 1, hidden_dim]

        # Get attention weights
        _, attn_weights = self.mha(query, key_value, key_value, need_weights=True)

        # attn_weights shape: [B, num_heads, 1, num_patches] if num_heads > 1
        # or [B, 1, num_patches] if num_heads == 1
        return attn_weights
