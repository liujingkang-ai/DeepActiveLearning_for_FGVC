import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

    
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        # print('part attention length:', length)
        last_map = x[0]
        # print('last_map shape:', last_map.shape)
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]    # 把 cls token去掉？
        # print('last_map shape after:', last_map.shape)

        _, max_inx = last_map.max(2)
        # print('max_inx.shape:', max_inx.shape)  # 12个head中最大关系的那个patch索引
        return _, max_inx


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels, out_channels):
        super(Embeddings, self).__init__()

        n_patches = img_size[0] * img_size[1]
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)
        
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, out_channels))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_channels))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        # print('self.patch_embeddings shape:', x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        # print('patch embedding after shape:', x.shape)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings    


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, img_size=(8, 8), in_channels=2048):

        super().__init__()

        self.embeddings = Embeddings(img_size=img_size, in_channels=in_channels, out_channels=d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.part_select = Part_Attention()
        self.part_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.part_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, return_attns=True):

        enc_output = self.dropout(self.embeddings(x))
        enc_output = self.layer_norm(enc_output)

        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        part_num, part_inx = self.part_select(enc_slf_attn_list)
        part_inx = part_inx + 1
        parts = []
        # 8, 12
        B, num = part_inx.shape
        # 找12个不同attention head中与结果cls最相关的12个patches
        for i in range(B):
            parts.append(enc_output[i, part_inx[i,:]])
        '''
        parts length: 8   size: torch.Size([12, 768])
        stack parts: torch.Size([8, 12, 768])
        hidden_states:([8, 197, 768])
        concat shape: torch.Size([8, 13, 768])
        '''
        # print('parts length:', len(parts), 'size:', parts[0].shape)
        parts = torch.stack(parts).squeeze(1)
        # print('stack parts:', parts.shape)
        concat = torch.cat((enc_output[:,0].unsqueeze(1), parts), dim=1)
        # print('concat shape:', concat.shape)
        part_states, part_weights = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)
        # print('part_encoded shape:', part_encoded.shape)
        
        return part_encoded


class MGMBTNet(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_word_vec=256, d_model=512, d_inner=1024,
            n_layers=4, n_head=8, d_k=64, d_v=64, dropout=0.1, img_size=(8, 8), in_channels=2048):

        super().__init__()

        # self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        
        self.encoder = Encoder(
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, img_size=img_size, in_channels=in_channels)


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        # self.part_head = nn.Linear(512, 100)
        

    def forward(self, x):
        enc_output = self.encoder(x)
        # out = self.part_head(enc_output[:, 0])
        
        return enc_output