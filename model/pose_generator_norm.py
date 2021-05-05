import torch
import torch.nn as nn
from model.audio_encoder import RNN
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from model.point_transformer_pytorch import PointTransformerLayer

#model for generator decoder
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
# use standard conv-relu-pool approach
class res_linear_layer(nn.Module):
    
    def __init__(self, linear_hidden = 1024,time=1024):
        super(res_linear_layer,self).__init__()
        self.layer = nn.Sequential(        
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU()
                      
        )
    def forward(self,input):
        output = self.layer(input)
        return output
        
class hr_pose_generator(nn.Module):
    def __init__(self,batch,hidden_channel_num=64,input_c = 266,linear_hidden = 1024):
        super(hr_pose_generator,self).__init__()
        self.batch=batch
        #self.relu = nn.ReLU()
        #self.decoder = nn.GRU(bidirectional=True,hidden_size=36, input_size=266,num_layers= 3, batch_first=True)
        #self.fc=nn.Linear(72,36)
        self.rnn_noise = nn.GRU( 10, 10, batch_first=True)
        self.rnn_noise_squashing = nn.Tanh()
        # state size. hidden_channel_num*8 x 360 x 640
        self.layer0 = nn.Linear(267,linear_hidden)
        #self.relu = nn.ReLU()
        #self.bn=nn.BatchNorm1d(50)
        self.layer1 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer2 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer3 = res_linear_layer(linear_hidden = linear_hidden)
        self.dropout =  nn.Dropout(p=0.5)
        self.final_linear1 = nn.Linear(linear_hidden,128)
        self.final_linear2 = nn.Linear(linear_hidden,3)
        self.final_linear3 = nn.Linear(128,36)

        self.first = nn.Linear(36,50)
        #self.sa = PAM_Module(50)
        #self.ca = CAM_Module(50)
        #self.ba = CAM_Module(16)
        self.attn = PointTransformerLayer(dim = 128, pos_mlp_hidden_dim = 64, attn_mlp_hidden_mult = 4)

    def forward(self, input, first_skeleton):
        #print(input.size()) [16, 50, 256]
        #print('first_skeleton', first_skeleton.size()) [16, 1, 36]
        first_skeleton = first_skeleton.view(self.batch,36) # [16, 36]
        first_skeleton = self.first(first_skeleton) #[16, 50]
        first_skeleton = first_skeleton.view(self.batch,50,1) #[16, 50, 1]
        #print('first_skeleton', first_skeleton.size()) 

        noise = torch.FloatTensor(self.batch, 50, 10).normal_(0, 0.33).cuda()
        #print(noise.size()) [16, 50, 10]
        aux, h = self.rnn_noise(noise)
        #print('aux', aux.size()) [16, 50, 10]
        aux = self.rnn_noise_squashing(aux)
        #print('aux', aux.size()) [16, 50, 10]
        input = torch.cat([input, aux, first_skeleton], 2) #[16, 50, 267]
        #print('input',input.size())
        input = input.view(-1,267)
        #print('input', input.size()) [800, 266]
        output = self.layer0(input)
        output = self.layer1(output) + output
        output = self.layer2(output) + output
        output = self.layer3(output) + output
        output = self.dropout(output)
        #print(output.size()) [800, 1024]
        output1 = self.final_linear1(output)#,36
        output2 = self.final_linear2(output)
        #print(output.size()) [800,36]
        feats = output1.view(self.batch,50,128)
        #print('feats',feats.size())
        #print('output', output.size())
        pos = output2.view(self.batch,50,3)
        #print('pos',pos.size())
        #output1 = self.ca(x)
        #output2 = self.sa(x)
        #x = x.permute(1,0,2,3)
        #output3 = self.ba(x).permute(1,0,2,3)
        #print('output3',output3.size())
        #output = output1 + output2 + output3 #bad results
        #output = output.view(self.batch,50,36)
        #print('self attention', output.size())
        #[16,  50, 36]
        mask = torch.ones(self.batch, 50).bool().cuda()
        #print('mask',mask.size())
        output = self.attn(feats, pos, mask = mask) # (self.batch, 50, 36)
        #print('output',output.size())
        output = self.final_linear3(output)
        #print('output2',output.size())
        output = self.rnn_noise_squashing(output)
        return output
    
        

class Generator(nn.Module):
    def __init__(self,batch):
        super(Generator,self).__init__()    
        self.audio_encoder=RNN(batch)
        self.pose_generator=hr_pose_generator(batch)
        self.batch=batch

    def forward(self,input, first_skeleton):
        #print('audio input', input.size()) #[50, 16batchsize, 1600]
        output=self.audio_encoder(input)#input 50,1,1600
        #print('audio output', output.size()) #[16, 50, 256]
        output=self.pose_generator(output, first_skeleton)#1，50，36
        #print('skeleton ouput', output.size()) #[16, 50, 36]
        return output#1,50,36
        
