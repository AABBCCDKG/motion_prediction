'''
Class UNetGenerator:
- __init__(self, input_channels: int, output_channels: int) 初始化U-Net生成器模型
- forward(self, x: torch.Tensor) -> torch.Tensor 前向传播，返回带有关键节点的草图

Class SharedConvLayers:
- __init__(self, input_channels: int) 初始化共享卷积层
- forward(self, x: torch.Tensor) -> torch.Tensor 前向传播，返回共享特征图

Class Image2Sketch:
save_dir_of_pickle: str 数据集的pickle文件保存路径
save_dir_of_model_image2sketch: str 生成器模型的保存路径
train_data: List[Tuple[torch.Tensor, torch.Tensor]] 训练数据集（图片和草图对）
val_data: List[Tuple[torch.Tensor, torch.Tensor]] 验证数据集（图片和草图对）
test_data: List[Tuple[torch.Tensor, torch.Tensor]] 测试数据集（图片和草图对）
generator: GeneratorModel 生成器模型
discriminator: DiscriminatorModel 多尺度判别器模型
cgan: cGANModel cGAN模型，联合生成器和判别器

- __init__(self, save_dir_of_pickle: str, shared_layers: SharedConvLayers) 初始化image2sketch模型，包含共享卷积层, 加载数据集
- load_data_from_pickle(self) 从pickle文件中加载数据，返回训练集，验证集和测试集
- build_generator(self) 构建生成器模型，生成器的输入是当前帧和前一帧，输出是带有关键节点的草图，使用U-Net结构（？）
- build_discriminator(self) 构建多尺度判别器模型，判别器的输入是当前帧，前一帧和生成器生成的草图
- build_cgan(self) 构建cGAN模型，将生成器和判别器连接起来
- forward(self, x: torch.Tensor) -> torch.Tensor 前向传播，返回生成的草图
- train(self, epochs: int, batch_size: int) 训练cGAN模型，定义并使用生成器和判别器的损失函数（二值交叉熵损失函数）
- evaluate(self) 在验证集上评估模型性能,使用Adam优化器(Adam是一种自适应学习旅优化算法，可以调整不同参数的学习率)
- test(self) 在测试集上评估模型性能，生成草图
- save(self, save_dir_of_model_cgan_image2sketch: str) 保存训练好的模型
- load_model(self, save_dir_of_model_cgan_image2sketch: str) 加载训练好的生成器模型
- generate_sketch(self, generated_frames_store_folder: str, generated_sketches_store_folder: str, input_images: torch.Tensor) -> torch.Tensor 使用训练好的生成器模型对输入图像生成草图

Class MotionPrediction:
labels_dir: str 标签数据的文件夹路径
sketch_dir: str 草图数据的文件夹路径
train_data: List[Tuple[torch.Tensor, torch.Tensor]] 训练数据集（输入草图序列，目标草图序列）
val_data: List[Tuple[torch.Tensor, torch.Tensor]] 验证数据集（输入草图序列，目标草图序列）
test_data: List[Tuple[torch.Tensor, torch.Tensor]] 测试数据集（输入草图序列，目标草图序列）
model: Seq2SeqModel Seq2Seq模型

- __init__(self, labels_dir: str, sketch_dir: str, shared_layers: SharedConvLayers) 初始化motion prediction模型，包含共享卷积层, 初始化labels_dir, sketch_dir
- generate_sketch_from_keypoints(self, keypoints: np.ndarray, sketch_dir: str) 
根据labels文件夹中的.mat文件内的关键点生成草图，依然是.mat文件，每个.mat文件包含一个视频所有帧的标签,并保存在sketch_dir文件夹中
- split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) 将sketch_dir中的.mat文件按照比例，划分数据集, 将数据集划分为训练集，验证集和测试集
- load_data(self) 从训练数据中，加载数据集，将每个.mat文件按比例拆分成输入草图序列和目标草图序列,而不是将不同的.mat文件分别作为输入和输出序列。随后调用splite_data函数拆分训练集，验证集和测试集，并分别储存为train_data, val_data和test_data,每个元素为(input_seq, target_seq)的元组
- build_model_of_seq2seq(self, input_shape: Tuple[int, int, int], seq_len_input: int, seq_len_output: int, filter: int, kernel_size: Tuple[int, int], activation: str) 构建Seq2Seq模型，需要用ConvLSTM做编码器和解码器，编码器(ConvLSTM)编码输入草图序列为隐藏状态，解码器(ConvLSTM)在编码器隐藏状态基础上生成目标草图序列
- train(self, epochs: int, batch_size: int) 训练Seq2Seq模型，使用模型的fit方法，输入为（输入草图序列，目标草图序列）,使用Huber Loss作为损失函数
- evaluate(self) 在验证集上评估模型性能，使用Adam优化器(Adam是一种自适应学习旅优化算法，可以调整不同参数的学习率)
- test(self)
- save(self, save_dir_of_model_seq2seq: str) 保存训练好的模型
- load_model(self, save_dir_of_model_seq2seq: str)
- predict(self, generated_sketches_store_folder: str, predicted_sketches_store_folder: str, input_sketches: torch.Tensor) -> torch.Tensor 使用训练好的模型进行预测目标草图序列。编码器编码输入草图序列为隐藏状态，解码器在隐藏状态基础上生成目标序列

Class Sketch2Image:
save_dir_of_pickle: str 数据集的pickle文件保存路径
save_dir_of_model_sketch2image: str 生成器模型的保存路径
train_data: List[Tuple[torch.Tensor, torch.Tensor]] 训练数据集（图片草图对）
val_data: List[Tuple[torch.Tensor, torch.Tensor]] 验证数据集（图片草图对）
test_data: List[Tuple[torch.Tensor, torch.Tensor]] 测试数据集（图片草图对）
generator: GeneratorModel 生成器模型
discriminator: DiscriminatorModel 多尺度判别器模型
cgan: cGANModel cGAN模型，联合生成器和判别器

- __init__(self, save_dir_of_pickle: str, shared_layers: SharedConvLayers)
- load_data_from_pickle(self): 加载预处理后的数据集,从pickle文件中加载数据，返回训练集，验证集和测试集
- build_generator(self) 构建生成器模型，生成器的输入是当前帧和前一帧，输出是带有关键节点的草图
- build_discriminator(self) 构建多尺度判别器模型，判别器的输入是当前帧，前一帧和生成器生成的草图
- build_cgan(self) 构建cGAN模型，将生成器和判别器连接起来
- forward(self, x: torch.Tensor) -> torch.Tensor 前向传播，返回生成的草图
- train(self, epochs: int, batch_size: int) 训练cGAN模型
- evaluate(self) 在验证集上评估模型性能,使用Adam优化器(Adam是一种自适应学习旅优化算法，可以调整不同参数的学习率)
- test(self)
- save(self, save_dir_model_sketch2image: str)
- load_model(self, save_dir_of_model_sketch2image: str)
- generate_image(self, predicted_sketches_store_folder: str, generated_image_folder: str, input_sketches: torch.Tensor) -> torch.Tensor 使用训练好的生成器模型对输入的草图和标签生成未来帧的图像

Class MultiTaskLearning:
- __init__(self, input_channels: int) 初始化多任务学习模型，包含共享卷积层
- forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] 前向传播，返回生成的草图、预测的草图和预测的关键点
'''

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import pickle
import scipy.io

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        print(f"编码前的形状: {x.shape}")  # 添加此行检查输入形状
        enc1 = self.encoder[0:2](x) 
        enc2 = self.encoder[2:5](enc1)
        enc3 = self.encoder[5:8](enc2)
        enc4 = self.encoder[8:](enc3)

        dec1 = self.decoder[0:3](enc4)
        dec2 = self.decoder[3:6](torch.cat([dec1, enc3], dim=1))
        dec3 = self.decoder[6:9](torch.cat([dec2, enc2], dim=1))
        dec4 = self.decoder[9:](torch.cat([dec3, enc1], dim=1))

        return dec4

# 多尺度判别器
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels: int):
        super(MultiScaleDiscriminator, self).__init__() # 初始化父类
        
        def discriminator_block(in_filters, out_filters, normalize=True): # 定义判别器块
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)] # 卷积层
            if normalize: # 是否使用归一化
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # 激活函数
            return layers # 返回层
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# Shared Convolutional Layers
class SharedConvLayers(nn.Module):
    def __init__(self, input_channels: int):
        super(SharedConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

# Image to Sketch Model
class Image2Sketch(nn.Module):
    def __init__(self, save_dir_of_pickle: str, shared_layers: SharedConvLayers):
        super(Image2Sketch, self).__init__()
        self.save_dir_of_pickle = save_dir_of_pickle
        self.shared_layers = shared_layers
        self.train_data, self.val_data, self.test_data = self.load_data_from_pickle()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.cgan = self.build_cgan()

    def load_data_from_pickle(self):
        with open(os.path.join(self.save_dir_of_pickle, 'train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(self.save_dir_of_pickle, 'val_data.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        with open(os.path.join(self.save_dir_of_pickle, 'test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        return train_data, val_data, test_data

    def build_generator(self):
        return UNetGenerator(input_channels=6, output_channels=1)  # 输入通道数为6，因为包含当前帧和前一帧，每帧3个通道

    def build_discriminator(self):
        return MultiScaleDiscriminator(input_channels=7)  # 输入通道数为7，因为包含当前帧、前一帧和生成的草图


    def build_cgan(self):
        class cGAN(nn.Module):
            def __init__(self, generator, discriminator):
                super(cGAN, self).__init__()
                self.generator = generator
                self.discriminator = discriminator

            def forward(self, x, y):
                fake_y = self.generator(x)
                real_output = self.discriminator(torch.cat((x, y), 1))
                fake_output = self.discriminator(torch.cat((x, fake_y), 1))
                return fake_y, real_output, fake_output

        return cGAN(self.generator, self.discriminator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_layers(x)
        return self.generator(shared_features)

    def train(self, epochs: int):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion_gan = nn.BCELoss()
        criterion_pixelwise = nn.L1Loss()

        for epoch in range(epochs):
            for batch in self.train_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    real_labels = torch.ones((inputs.size(0), 1))
                    fake_labels = torch.zeros((inputs.size(0), 1))

                    optimizer_g.zero_grad()
                    fake_targets, real_output, fake_output = self.cgan(inputs, targets)
                    loss_g = criterion_gan(fake_output, real_labels) + criterion_pixelwise(fake_targets, targets)
                    loss_g.backward()
                    optimizer_g.step()

                    optimizer_d.zero_grad()
                    loss_d_real = criterion_gan(real_output, real_labels)
                    loss_d_fake = criterion_gan(fake_output, fake_labels)
                    loss_d = (loss_d_real + loss_d_fake) / 2
                    loss_d.backward()
                    optimizer_d.step()

    def evaluate(self):
        self.eval()
        criterion_pixelwise = nn.L1Loss()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    fake_targets = self.forward(inputs)
                    val_loss += criterion_pixelwise(fake_targets, targets).item()
        return val_loss / len(self.val_data)

    def test(self):
        self.eval()
        criterion_pixelwise = nn.L1Loss()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    fake_targets = self.forward(inputs)
                    test_loss += criterion_pixelwise(fake_targets, targets).item()
        return test_loss / len(self.test_data)

    def save(self, save_dir_of_model_cgan_image2sketch: str):
        torch.save(self.cgan.state_dict(), os.path.join(save_dir_of_model_cgan_image2sketch, 'cgan.pth'))

    def load_model(self, save_dir_of_model_cgan_image2sketch: str):
        self.cgan.load_state_dict(torch.load(os.path.join(save_dir_of_model_cgan_image2sketch, 'cgan.pth')))

    def generate_sketch(self, generated_frames_store_folder: str, generated_sketches_store_folder: str, input_images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            sketches = self.forward(input_images)
            for i, sketch in enumerate(sketches):
                save_path = os.path.join(generated_sketches_store_folder, f'sketch_{i}.png')
                cv2.imwrite(save_path, sketch.cpu().numpy() * 255)
        return sketches

# Motion Prediction Model
class MotionPrediction(nn.Module):
    def __init__(self, labels_dir: str, sketch_dir: str, shared_layers: SharedConvLayers):
        super(MotionPrediction, self).__init__()
        self.labels_dir = labels_dir
        self.sketch_dir = sketch_dir
        self.shared_layers = shared_layers
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.model = self.build_model_of_seq2seq((256, 256, 3), 10, 10, 64, (3, 3), 'relu')

    def generate_sketch_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        num_frames = keypoints.shape[0]
        sketch_frames = []

        for frame_idx in range(num_frames):
            sketch = np.zeros((256, 256), dtype=np.uint8)
            for i in range(len(keypoints[frame_idx]) - 1):
                x1, y1 = keypoints[frame_idx, i]
                x2, y2 = keypoints[frame_idx, i + 1]
                cv2.line(sketch, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
            sketch_frames.append(sketch)
        
        return sketch_frames

    def split_data(self, train_ratio: float, val_ratio: float, test_ratio: float):
        train_val_data, self.test_data = train_test_split(self.data, test_size=test_ratio)
        self.train_data, self.val_data = train_test_split(train_val_data, test_size=val_ratio / (train_ratio + val_ratio))

    def load_data(self):
        data = []
        for filename in os.listdir(self.labels_dir):
            if filename.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(self.labels_dir, filename))
                x_coords = mat['x']
                y_coords = mat['y']
                keypoints = np.stack((x_coords, y_coords), axis=-1)

                sketches = self.generate_sketch_from_keypoints(keypoints)
                
                for frame_idx, sketch in enumerate(sketches):
                    cv2.imwrite(os.path.join(self.sketch_dir, f'sketch_{filename[:-4]}_{frame_idx}.png'), sketch)

                for i in range(keypoints.shape[1] - 1):
                    input_seq = (x_coords[:, i], y_coords[:, i])
                    target_seq = (x_coords[:, i + 1], y_coords[:, i + 1])
                    data.append((input_seq, target_seq))

        self.data = data
        self.split_data(0.7, 0.2, 0.1)
        return self.train_data, self.val_data, self.test_data

    def build_model_of_seq2seq(self, input_shape: Tuple[int, int, int], seq_len_input: int, seq_len_output: int, filter: int, kernel_size: Tuple[int, int], activation: str):
        class ConvLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
                super(ConvLSTM, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.kernel_size = kernel_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

            def forward(self, x):
                h, _ = self.lstm(x)
                return h

        encoder = ConvLSTM(input_shape[0], filter, kernel_size, 1)
        decoder = ConvLSTM(filter, input_shape[0], kernel_size, 1)
        return nn.Sequential(encoder, nn.ReLU(inplace=True), decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_layers(x)
        return self.model(shared_features)

    def train(self, epochs: int):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.HuberLoss()

        for epoch in range(epochs):
            for batch in self.train_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

    def evaluate(self):
        self.eval()
        criterion = nn.HuberLoss()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    outputs = self.forward(inputs)
                    val_loss += criterion(outputs, targets).item()
        return val_loss / len(self.val_data)

    def test(self):
        self.eval()
        criterion = nn.HuberLoss()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    outputs = self.forward(inputs)
                    test_loss += criterion(outputs, targets).item()
        return test_loss / len(self.test_data)

    def save(self, save_dir_of_model_seq2seq: str):
        torch.save(self.model.state_dict(), os.path.join(save_dir_of_model_seq2seq, 'seq2seq.pth'))

    def load_model(self, save_dir_of_model_seq2seq: str):
        self.model.load_state_dict(torch.load(os.path.join(save_dir_of_model_seq2seq, 'seq2seq.pth')))

    def predict(self, generated_sketches_store_folder: str, predicted_sketches_store_folder: str, input_sketches: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            predictions = self.forward(input_sketches)
            for i, prediction in enumerate(predictions):
                save_path = os.path.join(predicted_sketches_store_folder, f'prediction_{i}.png')
                cv2.imwrite(save_path, prediction.cpu().numpy() * 255)
        return predictions

# Sketch to Image Model
class Sketch2Image(nn.Module):
    def __init__(self, save_dir_of_pickle: str, shared_layers: SharedConvLayers):
        super(Sketch2Image, self).__init__()
        print("初始化 Sketch2Image 模型...")
        self.save_dir_of_pickle = save_dir_of_pickle
        self.shared_layers = shared_layers
        
        print("从 pickle 文件加载数据...")
        self.train_data, self.val_data, self.test_data = self.load_data_from_pickle()
        print("数据加载完成。")

        print("构建生成器...")
        self.generator = self.build_generator()
        print("生成器构建完成。")

        print("构建判别器...")
        self.discriminator = self.build_discriminator()
        print("判别器构建完成。")

        print("构建 cGAN...")
        self.cgan = self.build_cgan()
        print("cGAN 构建完成。")
        
        print("Sketch2Image 模型初始化完成。")

    def load_data_from_pickle(self):
        with open(os.path.join(self.save_dir_of_pickle, 'train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(self.save_dir_of_pickle, 'val_data.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        with open(os.path.join(self.save_dir_of_pickle, 'test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        return train_data, val_data, test_data

    def build_generator(self):
        return UNetGenerator(input_channels=6, output_channels=1)  # 输入通道数为6，因为包含当前帧和前一帧，每帧3个通道

    def build_discriminator(self):
        return MultiScaleDiscriminator(input_channels=7) # 输入通道数为7，因为包含当前帧、前一帧和生成的草图

    def build_cgan(self):
        class cGAN(nn.Module):
            def __init__(self, generator, discriminator):
                super(cGAN, self).__init__()
                self.generator = generator
                self.discriminator = discriminator

            def forward(self, x, y):
                fake_y = self.generator(x)
                real_output = self.discriminator(torch.cat((x, y), 1))
                fake_output = self.discriminator(torch.cat((x, fake_y), 1))
                return fake_y, real_output, fake_output

        return cGAN(self.generator, self.discriminator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_layers(x)
        return self.generator(shared_features)

    def train(self, epochs: int):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion_gan = nn.BCELoss()
        criterion_pixelwise = nn.L1Loss()

        for epoch in range(epochs):
            for batch in self.train_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    real_labels = torch.ones((inputs.size(0), 1))
                    fake_labels = torch.zeros((inputs.size(0), 1))

                    optimizer_g.zero_grad()
                    fake_targets, real_output, fake_output = self.cgan(inputs, targets)
                    loss_g = criterion_gan(fake_output, real_labels) + criterion_pixelwise(fake_targets, targets)
                    loss_g.backward()
                    optimizer_g.step()

                    optimizer_d.zero_grad()
                    loss_d_real = criterion_gan(real_output, real_labels)
                    loss_d_fake = criterion_gan(fake_output, fake_labels)
                    loss_d = (loss_d_real + loss_d_fake) / 2
                    loss_d.backward()
                    optimizer_d.step()

    def evaluate(self):
        self.eval()
        criterion_pixelwise = nn.L1Loss()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    fake_targets = self.forward(inputs)
                    val_loss += criterion_pixelwise(fake_targets, targets).item()
        return val_loss / len(self.val_data)

    def test(self):
        self.eval()
        criterion_pixelwise = nn.L1Loss()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_data:
                for inputs, targets in zip(batch[::2], batch[1::2]):
                    fake_targets = self.forward(inputs)
                    test_loss += criterion_pixelwise(fake_targets, targets).item()
        return test_loss / len(self.test_data)

    def save(self, save_dir_of_model_sketch2image: str):
        torch.save(self.cgan.state_dict(), os.path.join(save_dir_of_model_sketch2image, 'cgan.pth'))

    def load_model(self, save_dir_of_model_sketch2image: str):
        self.cgan.load_state_dict(torch.load(os.path.join(save_dir_of_model_sketch2image, 'cgan.pth')))

    def generate_image(self, predicted_sketches_store_folder: str, generated_image_folder: str, input_sketches: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            images = self.forward(input_sketches)
            for i, image in enumerate(images):
                save_path = os.path.join(generated_image_folder, f'image_{i}.png')
                cv2.imwrite(save_path, image.cpu().numpy() * 255)
        return images

# Multi-Task Learning Model
class MultiTaskLearning(nn.Module):
    def __init__(self, shared_layers, image2sketch, motion_prediction, sketch2image):
        super(MultiTaskLearning, self).__init__()
        self.shared_layers = shared_layers
        self.image2sketch = image2sketch
        self.motion_prediction = motion_prediction
        self.sketch2image = sketch2image

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.shared_layers(x)
        sketch = self.image2sketch(shared_features)
        motion = self.motion_prediction(shared_features)
        image = self.sketch2image(shared_features)
        return sketch, motion, image
