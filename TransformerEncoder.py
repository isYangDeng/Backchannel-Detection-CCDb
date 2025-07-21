import torch
import torch.nn as nn
import torch.utils.data 
import math
import matplotlib as plt
s = nn.Softmax(dim=1)
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

        # [B, L, H, D/H]
        B, Lq, _ = Q.shape
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)
class CrossTransformerEncoder(nn.Module):
    def __init__(self, max_len, num_layers, input_dim_v, input_dim_a, num_heads, dim_feedforward):
        super().__init__()
        self.visual_encoder = nn.ModuleList([
            EncoderBlock(input_dim_v, num_heads, dim_feedforward) for _ in range(num_layers)
        ])
        self.acoustic_encoder = nn.ModuleList([
            EncoderBlock(input_dim_a, num_heads, dim_feedforward) for _ in range(num_layers)
        ])
        self.cross_attn_v2a = CrossAttention(query_dim=input_dim_a, context_dim=input_dim_v, embed_dim=input_dim_a, num_heads=num_heads)
        self.cross_attn_a2v = CrossAttention(query_dim=input_dim_v, context_dim=input_dim_a, embed_dim=input_dim_v, num_heads=num_heads)
        self.fc = nn.Linear(input_dim_a + input_dim_v, 1)

    def forward(self, visual_x, acoustic_x):
        for l in self.visual_encoder:
            visual_x = l(visual_x)
        for l in self.acoustic_encoder:
            acoustic_x = l(acoustic_x)

        # cross attention
        visual_x = visual_x + self.cross_attn_a2v(visual_x, acoustic_x)
        acoustic_x = acoustic_x + self.cross_attn_v2a(acoustic_x, visual_x)

        # concat features
        fused = torch.cat([visual_x.mean(dim=1), acoustic_x.mean(dim=1)], dim=-1)
        return fused, torch.sigmoid(self.fc(fused)).unsqueeze(1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, add_positional_encoding=True, max_len=300):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        self.add_positional_encoding = add_positional_encoding
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / (d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        if self.add_positional_encoding:
            #x = x + self.pe[:, :x.size(1)]
            self.pe = self.pe[:x.size(0), :]

        return x
def scaled_dot_product(q, k, v, mask=None): #dot product attention 计算a1 and ai 的relevance
    d_k = q.size()[-1]
   
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

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.2):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.input_dim = input_dim
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):

    def __init__(self, max_len, num_layers, add_positional_encoding=True, use_avg_pool=False, use_2_FC_for_avg_pool=False, **block_args):
        super().__init__()
        print("Creating Transformer class [TransformerEncoder]")
        self.use_avg_pool = use_avg_pool
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(d_model=block_args["input_dim"], add_positional_encoding=add_positional_encoding, max_len=max_len+1) # "+1" is added because of torch.ones(x.shape[0],1,x.shape[2]) in forward layer of TransformerEncoder (Was originally there)
        self.avg_pool = nn.AvgPool1d(max_len) # Only used if use_avg_pool = True
        if use_avg_pool and use_2_FC_for_avg_pool:
            self.fc = nn.Sequential(nn.Linear(block_args["input_dim"], 4*block_args["input_dim"]), 
                                    #  nn.ReLU(),
                                    #  nn.Dropout(p=drop_prob),
                                    nn.Linear(4*block_args["input_dim"], 1)) #classifier
        else:
            self.fc = nn.Linear(block_args["input_dim"], 1)
    def forward(self, x, mask=None):
        X0 = torch.ones(x.shape[0],1,x.shape[2]).to(torch.device('cuda')) #torch.Size([32, 1, 700])
        
        x = torch.cat((X0,x),1) #torch.Size([32, 301, 700])

        x = self.positional_encoding(x) #torch.Size([32, 301, 700])
        # print(f'x shape{x.shape}')
        for l in self.layers:
            x = l(x, mask=mask)
        
        if self.use_avg_pool:
            x_for_fc = self.avg_pool(x.permute(0,2,1)).permute(0,2,1) # Average on sequence. Original X: [batch, seq, dimension] -> changed to [batch, dim, seq] then avg -> changed back to original X [batch, 1, dim]
        else:
            x_for_fc = x
        
        return x,torch.sigmoid(self.fc(x_for_fc))

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        # print(len(self.layers))
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            # attention_maps.append(attn_map)
            x = l(x)
        return attn_map
        # return 1

