import math
import torch
import torch.nn as nn
import torch.utils.data


s = nn.Softmax(dim=-1)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.kv_proj = nn.Linear(context_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context, mask=None):
        # query: [B, Lq, Dq], context: [B, Lc, Dc]
        Q = self.q_proj(query)
        KV = self.kv_proj(context)
        K, V = KV.chunk(2, dim=-1)

        B, Lq, _ = Q.shape
        Lc = K.shape[1]

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, Lq, Dh]
        K = K.view(B, Lc, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, Lc, Dh]
        V = V.view(B, Lc, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, Lc, Dh]

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, add_positional_encoding=True, max_len=300):
        super().__init__()
        self.add_positional_encoding = add_positional_encoding

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        if self.add_positional_encoding:
            x = x + self.pe[:, :x.size(1), :]
        return x


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)

    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = s(attn_logits)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, L, 3*Dh]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [B, L, H, Dh]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim

        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, max_len, num_layers, input_dim, num_heads, dim_feedforward,
                 add_positional_encoding=True, dropout=0.2):
        super().__init__()
        print("Creating Transformer class [TransformerEncoder]")

        self.layers = nn.ModuleList([
            EncoderBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.positional_encoding = PositionalEncoding(
            d_model=input_dim,
            add_positional_encoding=add_positional_encoding,
            max_len=max_len
        )

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x: [B, T, D]
        x = self.positional_encoding(x)

        for l in self.layers:
            x = l(x, mask=mask)

        pooled = x.mean(dim=1)              # [B, D]
        probs = torch.sigmoid(self.fc(pooled))   # [B, 1]

        return pooled, probs

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        x = self.positional_encoding(x)

        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x, mask=mask)

        return attention_maps


class CrossTransformerEncoder(nn.Module):
    def __init__(self, max_len, num_layers, input_dim_v, input_dim_a, num_heads, dim_feedforward, dropout=0.2):
        super().__init__()

        self.visual_positional_encoding = PositionalEncoding(
            d_model=input_dim_v,
            add_positional_encoding=True,
            max_len=max_len
        )
        self.acoustic_positional_encoding = PositionalEncoding(
            d_model=input_dim_a,
            add_positional_encoding=True,
            max_len=max_len
        )

        self.visual_encoder = nn.ModuleList([
            EncoderBlock(
                input_dim=input_dim_v,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.acoustic_encoder = nn.ModuleList([
            EncoderBlock(
                input_dim=input_dim_a,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.cross_attn_v2a = CrossAttention(
            query_dim=input_dim_a,
            context_dim=input_dim_v,
            embed_dim=input_dim_a,
            num_heads=num_heads
        )
        self.cross_attn_a2v = CrossAttention(
            query_dim=input_dim_v,
            context_dim=input_dim_a,
            embed_dim=input_dim_v,
            num_heads=num_heads
        )

        self.fc = nn.Linear(input_dim_a + input_dim_v, 1)

    def forward(self, visual_x, acoustic_x):
        # visual_x: [B, T, 710]
        # acoustic_x: [B, T, 90]

        visual_x = self.visual_positional_encoding(visual_x)
        acoustic_x = self.acoustic_positional_encoding(acoustic_x)

        for l in self.visual_encoder:
            visual_x = l(visual_x)

        for l in self.acoustic_encoder:
            acoustic_x = l(acoustic_x)

        visual_x = visual_x + self.cross_attn_a2v(visual_x, acoustic_x)
        acoustic_x = acoustic_x + self.cross_attn_v2a(acoustic_x, visual_x)

        fused = torch.cat([visual_x.mean(dim=1), acoustic_x.mean(dim=1)], dim=-1)  # [B, 800]
        probs = torch.sigmoid(self.fc(fused))  # [B, 1]

        return fused, probs