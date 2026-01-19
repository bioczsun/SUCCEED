import torch
import torch.nn as nn

import hic.model.blocks as blocks

class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, seq_channels = 4, mid_hidden = 256, preencoded_seq = False):
        super(ConvModel, self).__init__()
        print('Initializing ConvModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, seq_channels = seq_channels, output_size = mid_hidden, num_blocks = 12, preencoded_seq = preencoded_seq)
        self.decoder = blocks.Decoder(mid_hidden * 2)

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        if isinstance(x, tuple):
            seq_inputs, epi_inputs = x
            seq_inputs = seq_inputs.float()
            epi_inputs = epi_inputs.float()
            x = self.encoder((seq_inputs, epi_inputs))
        else:
            x = self.move_feature_forward(x).float()
            x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to: 
        bs, feat, img_len
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map



class ConvTransModel(ConvModel):
    
    def __init__(self, num_genomic_features, seq_channels = 4, mid_hidden = 256, record_attn = False, succeed=True, preencoded_seq = False):
        super(ConvTransModel, self).__init__(num_genomic_features, seq_channels = seq_channels, mid_hidden = mid_hidden, preencoded_seq = preencoded_seq)
        print('Initializing ConvTransModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, seq_channels = seq_channels, output_size = mid_hidden, num_blocks = 12,succeed=succeed, preencoded_seq = preencoded_seq)
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        self.decoder = blocks.Decoder(mid_hidden * 2)
        self.record_attn = record_attn
        self.succeed = succeed
    
    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        if isinstance(x, tuple):
            seq_inputs, epi_inputs = x
            seq_inputs = seq_inputs.float()
            epi_inputs = epi_inputs.float()
            encoder_inputs = (seq_inputs, epi_inputs)
            if self.succeed:
                x = self.encoder(encoder_inputs)
            else:
                x = self.encoder(encoder_inputs)
                x = self.diagonalize(x)
                x = self.decoder(x).squeeze(1)
                if self.record_attn:
                    return x, attn_weights
                else:
                    return x
            x = self.diagonalize(x)
            x = self.decoder(x).squeeze(1)
            return x
        x = self.move_feature_forward(x.float())

        if self.succeed:
            x = self.encoder(x)
            x = self.diagonalize(x)
            x = self.decoder(x).squeeze(1)
            return x
        else:
            x = self.encoder(x)
            x = self.move_feature_forward(x)
            if self.record_attn:
                x, attn_weights = self.attn(x)
            else:
                x = self.attn(x)
            x = self.move_feature_forward(x)
            x = self.diagonalize(x)
            x = self.decoder(x).squeeze(1)
            if self.record_attn:
                return x, attn_weights
            else:
                return x

if __name__ == '__main__':
    main()
