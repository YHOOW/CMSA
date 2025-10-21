from torch import nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)




class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=64):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        context = torch.bmm(attention, v)

        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=64, num_heads=8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, key, value, query):

        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(num_heads, -1, dim_per_head)
        value = value.view( num_heads, -1, dim_per_head)
        query = query.view(num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.view(-1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=64, ffn_dim=512, dropout=0.2):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, x):
        output = x
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):

        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class Encoder1(nn.Module):
    def __init__(self,
               num_size,
               num_layers=1,
               model_dim=64,
               num_heads=8,
               ffn_dim=512,
               dropout=0.2):
        super(Encoder1, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Linear(256, model_dim)

        self.model_dim = model_dim

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]

        if self.seq_embedding.in_features != seq_len:
            self.seq_embedding = nn.Linear(seq_len, self.model_dim).to(inputs.device)

        output = self.seq_embedding(inputs)

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)

        return output, attentions

class CMSA(nn.Module):
    def __init__(self, class_num , view, input_size, low_feature_dim, high_feature_dim, device):
        super(CMSA, self).__init__()
        self.encoder = Encoder(input_size, low_feature_dim).to(device)
        self.decoder = Decoder(input_size, low_feature_dim).to(device)

        self.encoder1 = Encoder1(num_size=256)
        self.l3 = nn.Linear(64,  high_feature_dim)
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim*view, high_feature_dim),
        )
        self.view = view
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=low_feature_dim*view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
    def forward(self, xs):
        zs = self.encoder(xs)
        xrs = self.decoder(zs)
        return  xrs, zs

    def SDF(self, xs):
        z = self.encoder(xs)  # 使用第一个（也是唯一的）编码器对输入进行编码
        commonz, S= self.TransformerEncoder(z)  # 增加一个维度以适应批处理
        commonz = normalize(self.Common_view(commonz), dim=1)
        return commonz,S

    def apply_multihead_attention(self, S):
        S = S.transpose(0, 1)
        CF, _ = self.encoder1(S)
        CF = F.softmax(self.l3(CF), dim=1)
        return CF