# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Calculate the custom redshift loss
def redshift_hsc_loss(z_spec, z_photo, gamma=0.15):
    """
    Custom HSC loss function for redshift prediction.
    """
    dz = (z_photo - z_spec) / (1 + z_spec)
    denominator = 1.0 + torch.square(dz / gamma)
    loss = 1 - 1.0 / denominator
    return torch.mean(loss)

# Set up patch embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=5, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Add learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed  # Add position embedding
        return x

# Set up spectrum patch embedding
class SpectrumPatchEmbed(nn.Module):
    def __init__(self, input_length=7783, patch_size=8, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.linear = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 1, 7783)
        B, C, L = x.shape
        x = x.squeeze(1)  # (B, L)

        # Truncate to fit patch size exactly
        trunc_len = self.num_patches * self.patch_size
        x = x[:, :trunc_len]

        x = x.view(B, self.num_patches, self.patch_size)  # (B, num_patches, patch_size)
        x = self.linear(x) + self.pos_embed  # (B, num_patches, embed_dim)
        return self.norm(x)

# Set up cross attention blocks
class CrossAttentionBlock(nn.Module):
    """Cross-attention between image and spectrum features"""
    def __init__(self, embed_dim=256, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        # Cross attention
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        return query

# Set up encoder
class TransformerEncoder1D(nn.Module):
    def __init__(self, embed_dim=256, depth=4, nhead=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.norm(x)

# Define a function for randomly masking
def random_mask(x, mask_ratio):
    """Improved masking with better randomization"""
    B, N, D = x.shape
    num_mask = int(mask_ratio * N)
    
    # Generate random indices for each batch
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    # Create mask
    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    ids_mask = ids_shuffle[:, :num_mask]
    mask.scatter_(1, ids_mask, True)
    
    # Apply mask
    x_masked = x.clone()
    x_masked[mask] = 0.0
    
    return x_masked, mask

# Initialize the model
class MMAE_ViT(nn.Module):
    def __init__(self, 
                 img_size=64, 
                 patch_size=8, 
                 spectrum_len=7783, 
                 spectrum_patch_size=8,
                 embed_dim=256, 
                 transformer_depth=4, 
                 nhead=8,
                 cross_attention_layers=2,
                 dropout=0.1,
                use_spec=True):
        super().__init__()

        # Embeddings
        self.img_embed = PatchEmbed(img_size, patch_size, in_chans=5, embed_dim=embed_dim)
        self.spec_embed = SpectrumPatchEmbed(spectrum_len, patch_size=spectrum_patch_size, embed_dim=embed_dim)

        # Encoders
        self.img_encoder = TransformerEncoder1D(embed_dim, depth=transformer_depth, nhead=nhead, dropout=dropout)
        self.spec_encoder = TransformerEncoder1D(embed_dim, depth=transformer_depth, nhead=nhead, dropout=dropout)

        # Cross-attention layers for fusion
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, nhead, dropout) 
            for _ in range(cross_attention_layers)
        ])

        # Attention pooling
        self.img_pool = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.spec_pool = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.cls_token_img = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cls_token_spec = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Decoders
        self.img_decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, (img_size // patch_size) ** 2 * patch_size * patch_size * 5)
        )

        self.spec_decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, spectrum_len)
        )

        # Store dimensions
        self.patch_size = patch_size
        self.img_size = img_size
        self.spectrum_len = spectrum_len
        self.spectrum_patch_size = spectrum_patch_size
        self.spec_num_patches = spectrum_len // spectrum_patch_size
        self.embed_dim = embed_dim

        self.apply(self._init_weights)

        # Redshift regression
        self.redshift_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # Set up the forward pass
    def forward(self, img, spec=None, img_mask_ratio=0.25, spec_mask_ratio=0.25, use_spec=True):
        B = img.shape[0]
    
        img_tokens = self.img_embed(img)
        
        if use_spec and spec is not None:
            spec_tokens = self.spec_embed(spec)
            spec_tokens_masked, spec_mask = random_mask(spec_tokens, spec_mask_ratio)
        else:
            # Create zero tokens for spec if spectra are missing
            spec_tokens_masked = torch.zeros(B, self.spec_num_patches, self.embed_dim, device=img.device)
            spec_mask = torch.ones(B, self.spec_num_patches, device=img.device)
    
        img_tokens_masked, img_mask = random_mask(img_tokens, img_mask_ratio)
    
        img_encoded = self.img_encoder(img_tokens_masked)
        spec_encoded = self.spec_encoder(spec_tokens_masked)
    
        # Cross-attention fusion
        img_fused = img_encoded
        spec_fused = spec_encoded
        for cross_attn in self.cross_attention_layers:
            img_temp = cross_attn(img_fused, spec_fused)
            spec_temp = cross_attn(spec_fused, img_fused)
            img_fused = img_temp
            spec_fused = spec_temp
    
        # Pooling and fusion
        cls_token_img = self.cls_token_img.expand(B, -1, -1)
        cls_token_spec = self.cls_token_spec.expand(B, -1, -1)
        img_global, _ = self.img_pool(cls_token_img, img_fused, img_fused)
        spec_global, _ = self.spec_pool(cls_token_spec, spec_fused, spec_fused)
        img_global = img_global.squeeze(1)
        spec_global = spec_global.squeeze(1)
        fused_representation = torch.cat([img_global, spec_global], dim=1)

        # Perform redshift prediction
        redshift_pred = self.redshift_head(fused_representation).squeeze(-1)
    
        img_recon_flat = self.img_decoder(fused_representation)
        spec_recon = self.spec_decoder(fused_representation)
        img_recon = self.unpatchify_image(img_recon_flat)
    
        return {
            'img_recon': img_recon,
            'spec_recon': spec_recon,
            'fused_representation': fused_representation,
            'img_mask': img_mask,
            'spec_mask': spec_mask,
            'redshift_pred': redshift_pred
        }


    def unpatchify_image(self, x):
        B = x.shape[0]
        num_patches = (self.img_size // self.patch_size) ** 2
        x = x.view(B, num_patches, 5, self.patch_size, self.patch_size)
        H = W = self.img_size // self.patch_size
        x = x.view(B, H, W, 5, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 5, self.img_size, self.img_size)
        return x

    # Optional function with equal weighting for all losses OR only redshift loss
    # For weighting options, see train.py
    def compute_loss(self, outputs, img_target, spec_target, redshift_target=None, redshift_only=False):
        """
        Compute losses; optionally only redshift loss.
        """
        redshift_pred = outputs.get('redshift_pred', None)

        redshift_loss = None
        if redshift_target is not None and redshift_pred is not None:
            # Use your custom redshift loss
            redshift_loss = redshift_hsc_loss(redshift_target, redshift_pred)

        if redshift_only:
            total_loss = redshift_loss
            img_loss = torch.tensor(0.0, device=redshift_pred.device if redshift_pred is not None else 'cpu')
            spec_loss = torch.tensor(0.0, device=redshift_pred.device if redshift_pred is not None else 'cpu')
        else:
            img_loss = F.mse_loss(outputs['img_recon'], img_target)
            spec_loss = F.mse_loss(outputs['spec_recon'], spec_target)
            total_loss = img_loss + spec_loss
            if redshift_loss is not None:
                total_loss += redshift_loss

        return {
            'total_loss': total_loss,
            'img_loss': img_loss,
            'spec_loss': spec_loss,
            'redshift_loss': redshift_loss
        }


# Example usage and training function
def create_model():
    model = MMAE_ViT(
        img_size=64,
        patch_size=8,
        spectrum_len=7783,
        spectrum_patch_size=8,
        embed_dim=256,
        transformer_depth=6,
        nhead=8,
        cross_attention_layers=2,
        dropout=0.1
    )
    return model

# Function for training a step without MLflow 
def train_step(model, img, spec, redshift, optimizer):
    model.train()
    optimizer.zero_grad()

    outputs = model(img, spec, img_mask_ratio=0.75, spec_mask_ratio=0.75)

    loss_dict = model.compute_loss(
        outputs,
        img_target=img,
        spec_target=spec.squeeze(1),
        redshift_target=redshift,
        redshift_only=True  # Only use redshift loss here
    )

    loss_dict['total_loss'].backward()
    optimizer.step()

    return loss_dict

# Model summary function
def model_summary(model, img_shape=(1, 5, 64, 64), spec_shape=(1, 1, 7783)):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_img = torch.randn(img_shape)
        dummy_spec = torch.randn(spec_shape)
        outputs = model(dummy_img, dummy_spec)
        
    print(f"Input image shape: {img_shape}")
    print(f"Input spectrum shape: {spec_shape}")
    print(f"Output image shape: {outputs['img_recon'].shape}")
    print(f"Output spectrum shape: {outputs['spec_recon'].shape}")
    print(f"Fused representation shape: {outputs['fused_representation'].shape}")

#if __name__ == "__main__":
#    # Create and test model
#    model = create_model()
#    model_summary(model)