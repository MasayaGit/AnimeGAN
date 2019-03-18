
# coding: utf-8

# In[1]:


# パッケージのインポート
import os
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd

from AnimeDataLoader import AnimeDataset
from netG import weights_init,Generator
from netD import weights_init,Discriminator


# In[2]:


# 設定
workers = 2
batch_size=30
n_epoch = 400
lr = 0.00002
beta1 = 0.5
outf = './resultDRAGANanime'
display_interval = 100

# 乱数のシード（種）を固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# In[3]:


#データセットを読み込む
dataset = AnimeDataset(transform=transforms.Compose([
                          transforms.RandomResizedCrop(128, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))

# 訓練データをセットしたデータローダーを作成する
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))

# 学習に使用するデバイスを得る。可能ならGPUを使用する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


# In[4]:


# 生成器G。ランダムベクトルから贋作画像を生成する
netG = Generator().to(device)
netG.apply(weights_init)    # weights_init関数で初期化
print(netG)


# In[5]:


# 識別器D。画像が、元画像か贋作画像かを識別する
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)


# In[6]:


criterion = nn.MSELoss()    # 損失関数は平均二乗誤差損失

# オプティマイザ−のセットアップ
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  # 識別器D用
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  # 生成器G用

fixed_noise = torch.randn(1, 1,batch_size, 128, device=device)  # 確認用の固定したノイズ


# In[7]:


# 学習のループ
for epoch in range(n_epoch):
    for itr, data in enumerate(dataloader):
        real_image = data.to(device)     # 元画像
        sample_size = real_image.size(0)    # 画像枚数
        #バッチサイズ×１２８のノイズを一つ
        noise = torch.randn(1, 1,sample_size, 128, device=device)   # 正規分布からノイズを生成
        
        real_target = torch.full((sample_size,), 1., device=device)     # 元画像に対する識別信号の目標値「1」
        fake_target = torch.full((sample_size,), 0., device=device)     # 贋作画像に対する識別信号の目標値「0」
        
        ############################
        # 識別器Dの更新
        ###########################
        netD.zero_grad()    # 勾配の初期化

        output = netD(real_image)   # 識別器Dで元画像に対する識別信号を出力
        
        errD_real = criterion(output, real_target)  # 元画像に対する識別信号の損失値
        D_x = output.mean().item()

        fake_image = netG(noise)    # 生成器Gでノイズから贋作画像を生成
        
        output = netD(fake_image.detach())  # 識別器Dで元画像に対する識別信号を出力
        errD_fake = criterion(output, fake_target)  # 贋作画像に対する識別信号の損失値
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake    # 識別器Dの全体の損失
        errD.backward()    # 誤差逆伝播
      
        
        
         ############################
        # DRAGAN
        ###########################
        X = data
        
        alpha = torch.tensor(np.random.random(size=X.shape)).float()

        interpolates =  alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size())))
        interpolates = Variable(interpolates, requires_grad=True)
        interpolates = interpolates.to(device)
        d_interpolates = netD(interpolates)
        
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=torch.ones(d_interpolates.size()).to(device),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.to(device)
        
        lambda_gp = 10
        
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        gradient_penalty.backward()
        #######################################################################
        
        optimizerD.step()   # Dのパラメーターを更新

        ############################
        # 生成器Gの更新
        ###########################
        netG.zero_grad()    # 勾配の初期化
        
        output = netD(fake_image)   # 更新した識別器Dで改めて贋作画像に対する識別信号を出力
        errG = criterion(output, real_target)   # 生成器Gの損失値。Dに贋作画像を元画像と誤認させたいため目標値は「1」
        errG.backward()     # 誤差逆伝播
        D_G_z2 = output.mean().item()

        optimizerG.step()   # Gのパラメータを更新

        if itr % display_interval == 0: 
            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                  .format(epoch + 1, n_epoch,
                          itr + 1, len(dataloader),
                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if epoch == 0 and itr == 0:     # 初回に元画像を保存する
            vutils.save_image(real_image, '{}/real_samples.png'.format(outf),
                              normalize=True, nrow=10)

    ############################
    # 確認用画像の生成
    ############################
    if (epoch + 1) % 1 == 0:   # 3エポックごとに確認用の贋作画像を生成する
        fake_image = netG(fixed_noise)  # 
        vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                          normalize=True, nrow=10)

    ############################
    # モデルの保存
    ############################
    if (epoch + 1) % 50 == 0:   # 50エポックごとにモデルを保存する
        torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))
        torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))

