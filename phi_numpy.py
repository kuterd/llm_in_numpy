"""
    Model: phi-2
    Dim: 2560
    Layers: 32
    Heads: 32
    Activation: gelu_new
    Num Tokens: 50265
    Rotary Dimension: 32
"""

import os
import math

from safetensors.numpy import save_file, load_file
from tqdm import tqdm
from tokenizers import Tokenizer
import numpy as np

class Embedding:
    """
        For given token ids get embedding vectors.
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def forward(self, x):
        stack = []
        for token in x:
            stack.append(self.matrix[token])
        # return will be in shape (seq_len, dim)
        return np.stack(stack)

class LayerNorm:
    """
        Apply Layer Normalization.
        http://arxiv.org/abs/1607.06450
    """
    def __init__(self, weight_dict, prefix):
        self.weight = weight_dict[f"{prefix}.weight"]
        self.bias = weight_dict[f"{prefix}.bias"]

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.var(x, axis=-1, keepdims=True)
        return self.weight * ((x - mean) / np.sqrt(std + 1e-05)) + self.bias

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)


class PartialRotaryEmbedding:
    """
    Rotary positional embeddings.
    http://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim, max_seq_len):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2.0) / dim))

        t = np.arange(max_seq_len)
        freqs = np.outer(t, inv_freq)
        # different from the paper but has the same effect.
        emb = np.concatenate([freqs, freqs], axis=-1)

        self.cos = np.cos(emb)
        self.sin = np.sin(emb)
    
    def forward(self, len):
        return (
            self.cos[:len],
            self.sin[:len]
        )

def apply_rotary_pos_emb(x, cos, sin):
    cos = np.expand_dims(cos, axis=0)
    sin = np.expand_dims(sin, axis=0)
    # Differenty from the paper, but has the same effect.
    return x * cos + rotate_half(x) * sin

class LinearLayer:
    """
        Simple linear layer.
    """
    def __init__(self, weight_dict, prefix):
        self.weight = weight_dict[f"{prefix}.weight"]
        self.bias = weight_dict[f"{prefix}.bias"]

    def forward(self, x):
        return x @ self.weight.T + self.bias

def gelu(x):
    """
        Guassian Error Linear Units(gelu)
        https://arxiv.org/abs/1606.08415
    
        Implementation is based on https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py NewGELUActivation
    """
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3.0))))

class MLP:
    """
        Multilayer Perceptron.
    """
    def __init__(self, weight_dict, prefix):
        self.fc1 = LinearLayer(weight_dict, f"{prefix}.fc1")
        self.fc2= LinearLayer(weight_dict, f"{prefix}.fc2")

    def forward(self, x):
        x = self.fc1.forward(x)

        x = gelu(x)
        x = self.fc2.forward(x)
        return x

def sofmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

class SelfAttention:
    """
        Self attion.
        http://arxiv.org/abs/1706.03762
    """
    def __init__(self, weight_dict, prefix, num_heads=32, rotary_dim=32):
        self.hidden_dim = weight_dict[f"{prefix}.k_proj.weight"].shape[0]
        self.head_dim = self.hidden_dim // num_heads
        self.num_heads = num_heads

        self.dense = LinearLayer(weight_dict, f"{prefix}.dense")
        self.k_proj = LinearLayer(weight_dict, f"{prefix}.k_proj")
        self.q_proj = LinearLayer(weight_dict, f"{prefix}.q_proj")
        self.v_proj = LinearLayer(weight_dict, f"{prefix}.v_proj")

        self.rotary_dim = rotary_dim
        self.rotary = PartialRotaryEmbedding(self.rotary_dim, self.hidden_dim)

    def forward(self, input):
        cos, sin = self.rotary.forward(input.shape[0])

        seq_len, dim = input.shape
        k = self.k_proj.forward(input)
        q = self.q_proj.forward(input)
        v = self.v_proj.forward(input)

        # Convert to (num_heads, seq_len, head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim).transpose(1,0,2)
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1,0,2)
        v = v.reshape(seq_len, self.num_heads, self.head_dim).transpose(1,0,2)

        # Phi-2 only applies rotary positional embeddings to a part of the vectors.
        query_rot, query_pass = (
            q[..., :self.rotary_dim],
            q[..., self.rotary_dim:],
        )
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin)
        q = np.concatenate([query_rot, query_pass], axis=-1)
        
        key_rot, key_pass = (
            k[..., :self.rotary_dim],
            k[..., self.rotary_dim:],
        )
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin)
        k = np.concatenate([key_rot, key_pass], axis=-1)

        # Compute the attention matrix, this is based on the attention formula form the Attention Is All You Need paper.
        att = (q.astype(np.float32) @ k.transpose(0,2,1).astype(np.float32)) / math.sqrt(self.head_dim)
        # att is in shape (num_heads, seq_len, seq_len).
        
        # Compute and apply the causal attention mask.
        mask = np.full((1, seq_len, seq_len), float("-inf"), dtype=np.float32)
        mask = np.triu(mask, k=1)

        att = att + mask
        att = sofmax(att)
        result = att @ v
    
        # Convert to (seq, num_heads, head_dim)
        result = result.transpose(1,0,2)

        # Convert to (seq, num_heads * head_dim = dim)
        result = result.reshape(seq_len, dim)
        return self.dense.forward(result)
    
class TransformerDecoderLayer:
    """
        Transformer Decoder layer based on Attention Is All You Need paper/
    """ 
    def __init__(self, weight_dict, prefix):
        self.ln = LayerNorm(weight_dict, f"{prefix}.input_layernorm")
        self.mlp = MLP(weight_dict, f"{prefix}.mlp")
        self.attention = SelfAttention(weight_dict, f"{prefix}.self_attn")

    def forward(self, input):
        x = self.ln.forward(input)
        attn_output = self.attention.forward(x)
        hidden = self.mlp.forward(x)
        return attn_output + hidden + input

class LMHead:
    def __init__(self, weight_dict):
        self.ln = LayerNorm(weight_dict, "model.final_layernorm")
        self.linear = LinearLayer(weight_dict, "lm_head") 

    def forward(self, x):
        x = self.ln.forward(x)
        return self.linear.forward(x)

class CausalLM:
    def __init__(self, embeddings, decoders, lm_head):
        self.decoders = decoders
        self.embeddings = embeddings
        self.lm_head = lm_head

    def forward(self, tokens, progress=False):
        x = self.embeddings.forward(tokens)
        
        iterator = tqdm(self.decoders) if progress else self.decoders 
        for decoder in iterator:
            x = decoder.forward(x)
        logits = self.lm_head.forward(x)
        return logits

def load_phi2_model(folder_path):
    def model_get(name):
        # Get parameter name @name from the model.
        if name in model1:
            return model1[name]
        else:
            return model2[name]
    model = {    
        **load_file(os.path.join(folder_path, "model-00001-of-00002.safetensors")),
        **load_file(os.path.join(folder_path, "model-00002-of-00002.safetensors"))
    }
    
    lm_head = LMHead(model)
    embeddings = Embedding(model["model.embed_tokens.weight"])

    decoders = []

    layers = 32
    prefix = "model.layers"
    for l in range(0, layers):
        decoders.append(TransformerDecoderLayer(model, f"{prefix}.{l}"))
    return CausalLM(embeddings, decoders, lm_head)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--progress", help="display a progress bar:", action="store_true")
parser.add_argument("--location", default="./phi-2", help="Location of the phi-2 model.", type=str) 

args = parser.parse_args()

model = load_phi2_model(args.location)
tokenizer = Tokenizer.from_file(os.path.join(args.location, "tokenizer.json")) 

while True:
    prompt = input(">")
    tokens = tokenizer.encode(prompt).ids

    END_OF_SEQUENCE = 50256
    while True:
        logits = model.forward(tokens, progress=args.progress)
        tok = logits[-1].argmax()
        tokens.append(tok)
        print(tokenizer.decode(tokens))

        if tok == END_OF_SEQUENCE:
            break

