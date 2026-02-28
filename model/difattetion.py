import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMSNorm: headwise or tokenwise normalization."""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            norm_x = norm_x * self.weight
        return norm_x

class DiffAttention(nn.Module):
    """
    Differential Attention:
    - Split heads into two halves
    - Compute differential attention: A1 - lambda * A2
    - Multiply by value V and normalize
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even"
        self.dim = dim
        self.num_heads = num_heads
        self.effective_heads = num_heads // 2
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda for differential attention
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_init = lambda_init

    def forward(self, x):
        """
        x: [B, N, dim]
        return: [B, N, dim]
        """
        B, N, _ = x.shape
        #print("DiffAttention input:", x.shape, "mean:", x.mean().item(), "std:", x.std().item())
        # 1. Linear projections
        q = self.q_proj(x).view(B, N, 2*self.effective_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, 2*self.effective_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.effective_heads, 2*self.head_dim)

        # 2. Transpose for attention calculation
        q = q.transpose(1, 2)  # [B, 2*effective_heads, N, head_dim]
        k = k.transpose(1, 2)  # [B, 2*effective_heads, N, head_dim]
        v = v.transpose(1, 2)  # [B, effective_heads, N, 2*head_dim]

        # 3. Scaled dot-product attention
        q = q * self.scale
        attn_scores = torch.matmul(q, k.transpose(-1, -2))  # [B, 2*H_eff, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 4. Reshape attention into two halves
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)  # [B, H_eff, 2, N, N]

        # 5. Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 6. Differential attention
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # [B, H_eff, N, N]

        # 7. Apply to values
        attn_output = torch.matmul(diff_attn, v)  # [B, H_eff, N, 2*head_dim]

        # 8. RMSNorm
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # 9. Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2*self.effective_heads*self.head_dim)

        # 10. Final linear projection
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)
        return x_out


if __name__ == "__main__":
    B, N, dim = 1, 16*16, 32
    x = torch.randn(B, N, dim).cuda()
    model = DiffAttention(dim=dim, num_heads=8).cuda()
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
