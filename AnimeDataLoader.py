
# coding: utf-8

# In[1]:


from PIL import Image
import torch
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from GetAnimeFileName import getFileNameList

class AnimeDataset(torch.utils.data.Dataset):  
  
    def __init__(self, transform=None):
        # 指定する場合は前処理クラスを受け取る
        self.transform = transform
        # 画像とラベルの一覧を保持するリスト
        self.images = []
        
        # ルートフォルダーパス
        root = "/home/GANTest/animeface-character-dataset/thumb/"
        # 訓練の場合と検証の場合でフォルダわけ
        # 画像を読み込むファイルパスを取得します。
        
        fileNameList = getFileNameList()
        
        for fileName in fileNameList:
            root_file_path = os.path.join(root, fileName)
            #画像一覧を取得する。
            file_images = os.listdir(root_file_path)
            # 1個のリストにする。
            for image in zip(file_images):
                if image[0] == "color.csv" or image[0] == "ignore" :
                    continue
                self.images.append(os.path.join(root_file_path, image[0]))
            
    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得します。
        image = self.images[index]
        
        # 画像ファイルパスから画像を読み込む。
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # 前処理（transform）がある場合は前処理をいれる。
        if self.transform is not None:
            image = self.transform(image)
        # 画像とラベルのペアを返却します。
        return image
        
    def __len__(self):
        return len(self.images)
