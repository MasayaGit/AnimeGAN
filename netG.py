
# coding: utf-8

# In[1]:


import torch.nn as nn

def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
    :param m: ニューラルネットワークを構成する層
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            # 畳み込み層の場合
        #重み
        #print("Conv")
        m.weight.data.normal_(0.0, 0.02)
        #バイアス
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:# 全結合層の場合
        #print("Linear")
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:     # バッチノーマライゼーションの場合
        #print("BatchNorm")
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Gen_ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Gen_ResBlock,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, tensor):
        r_tensor = tensor
        output = self.conv_1(tensor)
        output = self.bn_1(output)
        output = self.relu(output)
        output = self.conv_2(output)
        output = self.bn_2(output)
        output += r_tensor
        return output


class PixelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()

    def forward(self, tensor):
        output = self.conv(tensor)
        output = self.pixel_shuffle(output)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Generator(nn.Module):
    def __init__(self,base=64):
        super(Generator,self).__init__()
        
        in_channels = 128
        self.dense_1 = nn.Linear(in_channels,base*16*16)
        self.bn_1 = nn.BatchNorm2d(base)
        self.relu_1 = nn.PReLU()
        
        self.r0 = Gen_ResBlock(base,base)
        self.r1 = Gen_ResBlock(base,base)
        self.r2 = Gen_ResBlock(base,base)
        
        self.bn_2 = nn.BatchNorm2d(base)
        self.relu_2 = nn.PReLU()
        
        self.p0 = PixelBlock(base,base*4)
        self.p1 = PixelBlock(base,base*4)
        self.p2 = PixelBlock(base,base*4)
        
        self.conv_1 = nn.Conv2d(base, 3, kernel_size=9, stride=1, padding=4, bias=True)
        self.tanh_1 = nn.Tanh()

    def forward(self, tensor):
        
        output = self.dense_1(tensor)
        output = output.view(-1, 64, 16, 16)
        output = self.bn_1(output)
        output = self.relu_1(output)
        r_output = output
        
        output = self.r0(output)
        output = self.r1(output)
        output = self.r2(output)
        
        output = self.bn_2(output)
        output = self.relu_2(output)
        output += r_output
        
        output = self.p0(output)
        output = self.p1(output)
        output = self.p2(output)
        
        output = self.conv_1(output)
        output = self.tanh_1(output)
        return output

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    gen = Generator()
    gen.apply(weights_init) 
    x = Variable(torch.rand((1, 128)), requires_grad=True)
    print(gen(x).shape)
