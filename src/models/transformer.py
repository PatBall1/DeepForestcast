# From https://github.com/MarcCoru/crop-type-mapping/blob/master/src/models/TransformerEncoder.py
# https://arxiv.org/pdf/1910.10536.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from models.ClassificationModel import ClassificationModel
from models.transformer.Models import Encoder



# See https://github.com/MarcCoru/crop-type-mapping

SEQUENCE_PADDINGS_VALUE=-1

# in_channels corresponds to bands (13*sentinel-2)?
# should correspond to input_dim[1] i.e. dynamic layers?

# n_layers corresponds to?

# n_classes - change to two?

# d_model and d_word_vec correstond to number of time steps?

# For the time series the time steps need to be encoded


class TransformerEncoder(ClassificationModel):
    def __init__(self, in_channels=13, len_max_seq=100,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, nclasses=6):

        self.d_model = d_model

        super(TransformerEncoder, self).__init__()

        self.inlayernorm = nn.LayerNorm(in_channels)
        self.convlayernorm = nn.LayerNorm(d_model)
        self.outlayernorm = nn.LayerNorm(d_model)

        self.inconv = torch.nn.Conv1d(in_channels, d_model, 1)

        self.encoder = Encoder(
            n_src_vocab=None, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.outlinear = nn.Linear(d_model, nclasses, bias=False)

        self.tempmaxpool = nn.MaxPool1d(int(len_max_seq))

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def _logits(self, x):
        # b,d,t - > b,t,d
        x = x.transpose(1,2)

        x = self.inlayernorm(x)

        # b,
        x = self.inconv(x.transpose(1,2)).transpose(1,2)

        x = self.convlayernorm(x)

        batchsize, seq, d = x.shape
        src_pos = torch.arange(1, seq + 1, dtype=torch.long).expand(batchsize, seq)

        if torch.cuda.is_available():
            src_pos = src_pos.cuda()

        enc_output, enc_slf_attn_list = self.encoder.forward(src_seq=x, src_pos=src_pos, return_attns=True)

        enc_output = self.outlayernorm(enc_output)

        enc_output = self.tempmaxpool(enc_output.transpose(1, 2)).squeeze(-1)

        logits = self.outlinear(enc_output)

        return logits, None, None, None

    def forward(self, x):

        logits, *_ = self._logits(x)

        logprobabilities = self.logsoftmax(logits)

        return logprobabilities, None, None, None

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot



# From https://github.com/pytorch/examples/blob/main/word_language_model/model.py

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
    
    
# https://github.com/justchenhao/BIT_CD/blob/master/models/networks.py

class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x    
    
    
    
# JB attempt
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Loosely based on DeepLSTM architecture
class CNNTransformer(torch.nn.Module):
    '''
    Model to convolve over temporally distinct input tensors and feed features
    (with positional encoding) through transformer
    '''
    def __init__(
        self,
        height=21,
        width=21,
        input_dim=(2, 5),
        hidden_dim=(16, 16, (16, 16), 8),
        kernel_size=((3, 3), (1, 3, 3), ((3, 3),), (3, 3)),
        num_layers=2,
        levels=(13,),
        dropout=0.2,
        bias=True,
        return_all_layers=False,
    ):
        super(DCNNTransformer, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
        )

        self.inconv = nn.Sequential(
            torch.nn.Conv3d(input_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
        )

        #REPLACE WITH TRANSFORMER
        #cell_input_size = height - 3 * (kernel_size[1][1] - 1)
        #
        #self.cell = ConvLSTM(
        #    input_size=(cell_input_size, cell_input_size),
        #    input_dim=hidden_dim[1],
        #    hidden_dim=hidden_dim[2],
        #    kernel_size=kernel_size[2],
        #    num_layers=num_layers,
        #    bias=bias,
        #    return_all_layers=return_all_layers,
        #)
        self.transformer = 

        self.final = nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[2][-1] + hidden_dim[0], hidden_dim[3], kernel_size[3]
            ),
            torch.nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[3]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[3] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv.forward(z)
        # Spatial pyramid pooling required on each branch
        z = spp_layer(z, self.levels)
        z = z.flatten()
        x = self.inconv.forward(x)
        #hidden, state = self.cell.forward(x)
        #x = hidden
        # Spatial pyramid pooling required on each branch
        x = spp_layer(x, self.levels)
        x = x.flatten()
        x = self.transformer.forward(x)
        # Join dynamic and static branches
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)
        return x.flatten()

