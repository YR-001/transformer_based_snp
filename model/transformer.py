import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import *



class TransformerSNP(nn.Module):

    def __init__(self,
                src_vocab_size,
                seq_len,
                embed_dim=512,
                num_blocks=6,
                expansion_factor=4,
                heads=8,
                dropout=0.2):
        super(TransformerSNP, self).__init__()

        self.encoder = EncoderBlock(seq_len=seq_len,
                               vocab_size=src_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        # additional layers for the regression task
        self.head = nn.Sequential(nn.Dropout(0.3),
                                nn.Linear(embed_dim, 128),
                                nn.Dropout(0.3),
                                nn.Linear(128, 64),
                                nn.Dropout(0.3),
                                nn.Linear(64, 1)) #regression task: num_classes=1

    def forward(self, source):
        
        # split the source to encode different parts of the chrobosome
        splitted_sources = torch.tensor_split(source, [218, 778, 1323, 1871], dim=1)

        # print('Shape of src',source.shape)
        # encode each part
        enc_out0 = self.encoder(splitted_sources[0])
        enc_out1 = self.encoder(splitted_sources[1])
        enc_out2 = self.encoder(splitted_sources[2])
        enc_out3 = self.encoder(splitted_sources[3])
        enc_out4 = self.encoder(splitted_sources[4])
        
        # print('Shape of enc_out1',enc_out0.shape)
        # print('Shape of enc_out2',enc_out1.shape)
        # print('Shape of enc_out3',enc_out2.shape)
        # print('Shape of enc_out4',enc_out3.shape)
        # print('Shape of enc_out5',enc_out4.shape)

        # concanate parts into one
        enc_out = torch.cat((enc_out0, enc_out1, enc_out2, enc_out3, enc_out4), 1)
        print('Shape enc_out', enc_out.shape)   #([batch, max_len, embed_dim]) #torch.Size([450, 220, 512])
        
        # use torch.mean to down size
        sequence_output = torch.mean(enc_out, dim=1) 
        print('Shape sequence_output', sequence_output.shape) #([batch, embed_dim]) #torch.Size([450, 512])


        result = self.head(sequence_output) 
        # print('Shape result',result.shape) #([batch, 1]) #torch.Size([450, 1])

        return result