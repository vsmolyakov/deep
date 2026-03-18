import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special='<|endoftext|>')
        for i in range(0, len(token_ids) - max_length, stride):
            input_seq = token_ids[i:i+max_length]
            target_seq = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_seq))
            self.target_ids.append(torch.tensor(target_seq))    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, float('-inf'))

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1]** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # compute context vector
        context_vec = attn_weights @ values 
        context_vec = context_vec.transpose(1, 2)  # (b, num_tokens, num_heads, head_dim) 

        # combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)  # (b, num_tokens, d_out)

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module): 
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim'])
        self.act = GELU()
        self.fc2 = nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
    def forward(self, x):
        x = nn.Sequential(self.fc1, self.act, self.fc2)(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.mha = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'], 
            num_heads=cfg['num_heads'],
            qkv_bias=cfg.get('qkv_bias')
        )
        self.ln1 = LayerNorm(cfg['emb_dim'])
        self.ln2 = LayerNorm(cfg['emb_dim'])
        self.ffn = FeedForward(cfg)
        self.drop = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)  # [batch_size, num_tokens, emb_size]
        x = self.drop(x)
        x = x + shortcut # add the original input back

        # shortcut connection for feed-forward block
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut # add the original input back

        return x



class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trfm_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trfm_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):
        # crop idx to the last `context_size` tokens
        idx_cond = idx[:, -context_size:]

        # get logits for the next token
        with torch.no_grad():
            logits = model(idx_cond)

        # focus on the last time step 
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # sample from the distribution to get the next token's index
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # append the new token id to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx 
    

def main():

    GPT_CONFIG_124M = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'drop_rate': 0.1,
        'qkv_bias': False
    }

    torch.manual_seed(123)
    model = GPT(GPT_CONFIG_124M)
    model.eval()  #disable dropout

    start_context = "The meaning of life is" 

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = generate_text_simple(
        model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M['context_length']
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    main()