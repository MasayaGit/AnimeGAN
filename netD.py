
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
        print("Conv")
        m.weight.data.normal_(0.0, 0.02)
        #バイアス
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:# 全結合層の場合
        print("Linear")
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    
    
    
    
class Dis_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dis_ResBlock, self).__init__()
        self.c0 =  nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.l =  nn.LeakyReLU(0.2)
        self.c1 =  nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
    #tensorはDiscriminatorのforwardで受け取る
    def forward(self, tensor):
        #LeakyReLUは出力のshapeは入力のshapeと同じ
        output = self.l(self.c0(tensor))
        output = self.c1(output)
        output = output + tensor
        output = self.l(output)
        return output

class Discriminator(nn.Module):
    def __init__(self,base=32):
        super(Discriminator, self).__init__()
        
        self.l  = nn.LeakyReLU(0.2)
        
        self.c0 = nn.Conv2d(3,base,kernel_size=4,stride=2,padding=1)
        self.r0 = Dis_ResBlock(base,base)
        self.r1 = Dis_ResBlock(base,base)
        self.c1 = nn.Conv2d(base,base*2,kernel_size=4,stride=2,padding=1)
        
        self.r2 = Dis_ResBlock(base*2,base*2)
        self.r3 = Dis_ResBlock(base*2,base*2)
        self.c2 = nn.Conv2d(base*2,base*4,kernel_size=4,stride=2,padding=1)
        
        self.r4 = Dis_ResBlock(base*4,base*4)
        self.r5 = Dis_ResBlock(base*4,base*4)
        self.c3 = nn.Conv2d(base*4,base*8,kernel_size=3,stride=2,padding=1)
        
        self.r6 = Dis_ResBlock(base*8,base*8)
        self.r7 = Dis_ResBlock(base*8,base*8)
        self.c4 = nn.Conv2d(base*8,base*16,kernel_size=3,stride=2,padding=1)
        
        self.r8 = Dis_ResBlock(base*16,base*16)
        self.r9 = Dis_ResBlock(base*16,base*16)
        self.c5 = nn.Conv2d(base*16,base*32,kernel_size=3,stride=2,padding=1)
        
        self.lastl = nn.Linear(256*4*4,1)

        
    def forward(self, tensor):
        #出力のshapeは入力のshapeと同じ
        output = self.l(self.c0(tensor))
        output = self.r0(output)
        output = self.r1(output)
        output = self.l(self.c1(output))
        output = self.r2(output)
        output = self.r3(output)
        output = self.l(self.c2(output))
        output = self.r4(output)
        output = self.r5(output)
        output = self.l(self.c3(output))
        output = self.r6(output)
        output = self.r7(output)
        output = self.l(self.c4(output))
        output = self.r8(output)
        output = self.r9(output)
        output = self.l(self.c5(output))
        output = output.view(output.size(0),-1)
        output = self.lastl(output)
        return output


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    dis = Discriminator()
    dis.apply(weights_init) 
    x = Variable(torch.rand((1, 3, 128, 128)), requires_grad=True)

