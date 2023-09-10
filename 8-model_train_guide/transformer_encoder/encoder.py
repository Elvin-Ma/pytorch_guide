import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(input_dim, hidden_dim, num_heads, dropout_rate) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, hidden_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForward(input_dim, hidden_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm(x + self.dropout(self.self_attention(x, x, x, mask)))
        x = self.layer_norm(x + self.dropout(self.feed_forward(x)))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        query = self.query(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        x = torch.matmul(self.dropout(attention_weights), value)
        x = x.transpose(1, 2).contiguous().view(batch_size, query_len, self.num_heads * self.head_dim)
        x = self.output_linear(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(self.dropout(F.relu(self.linear1(x))))
        x = self.linear2(x) + residual
        return x

if __name__ == "__main__":
    # 示例用法
    input_dim = 512
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    dropout_rate = 0.1
    batch_size = 10
    sequence_length = 20

    # 创建Transformer编码器实例
    encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout_rate)

    # 创建输入张量
    x = torch.randn(batch_size, sequence_length, input_dim)

    # 创建掩码（假设有掩盖的位置为0）
    mask = torch.ones(batch_size, sequence_length).byte()
    mask[5:, 15:] = 0  # 假设对序列位置5到10进行掩盖

    # 前向传播
    output = encoder(x, mask=mask)
    torch.onnx.export(encoder.eval(), (x, mask), "self_bert.onnx")

    # print(output.size())  # 输出形状：torch.Size([10, 20, 512])
    print("run encoder.py successfully !!!")