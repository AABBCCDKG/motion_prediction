'''
train.py: 训练模型

1. 导入数据处理，模型构建，模型训练，模型评估的所需要的库

2. 数据预处理：
2.1 使用DataPreprocessing类加载并处理数据集，包括加载视频帧和标签，调整大小，划分数据集，并保存预处理后的数据
2.2 初始化DataPreprocessing类
2.2.1 设置帧和标签的文件夹路径
2.2.2 调用load_data方法加载数据
2.2.3 调用resize_frame方法调整帧的大小
2.2.4 调用split_data方法划分数据集
2.2.5 调用save_data方法保存预处理后的数据

3. 图片到草图的转换：
3.1 加载数据
3.1.1 使用Image2Sketch类加载预处理后的训练集，验证集和测试集
3.2 构建和训练Image2Sketch网络
3.2.1 初始化Image2Sketch类，设置输入形状
3.2.2 调用build_generator()方法构建生成器模型
3.2.3 调用build_discriminator()方法构建多尺度判别器模型
3.2.4 调用build_cgan()法构建cGAN模型，将生成器和判别器连接起来
3.2.5 调用train()方法，传入训练数据，验证数据，训练轮次和批次大小，训练cGAN模型
3.2.6 调用evaluate()方法，在验证集上评估模型性能
3.2.7 调用test()方法，在测试集上评估模型性能，生成草图
3.2.8 调用save()方法，保存训练好的模型

4. 动作预测：
4.1 加载数据
4.1.1 使用MotionPrediction类加载预处理后的训练集，验证集和测试集

4.2 构建和训练MotionPrediction网络
4.2.1 初始化MotionPrediction类，设置输入形状
4.2.2 调用build_model()方法构建Seq2Seq模型，使用ConvLSTM做编码器和解码器
4.2.3 迁移学习
4.2.3.1 使用cGAN的预训练模型来初始化Seq2Seq模型的编码器和解码器
4.2.3.2 调用train()方法，传入训练数据，验证数据，训练轮次和批次大小，训练Seq2Seq模型
4.2.4 调用evaluate()方法，在验证集上评估模型性能
4.2.5 调用test()方法，在测试集上评估模型性能，生成未来帧的草图
4.2.6 调用save()方法，保存训练好的模型

5. 图片到草图的转换：
5.1 加载数据
5.1.1 使用Image2Sketch类加载预处理后的训练集，验证集和测试集

5.2 构建和训练Image2Sketch网络
5.2.1 初始化Image2Sketch类，设置输入形状
5.2.2 调用build_generator()方法构建生成器模型
5.2.3 调用build_discriminator()方法构建多尺度判别器模型
5.2.4 调用build_cgan()法构建cGAN模型，将生成器和判别器连接起来
5.2.5 迁移学习
5.2.5.1 使用cGAN(Image2sketch)和Seq2Seq的预训练模型来初始化cGAN(Sketch2Image)模型的生成器和判别器
5.2.5.2 调用train()方法，传入训练数据，验证数据，训练轮次和批次大小，训练cGAN模型
5.2.6 调用evaluate()方法，在验证集上评估模型性能
5.2.7 调用test()方法，在测试集上评估模型性能，生成草图
5.2.8 调用save()方法，保存训练好的模型

6. 多任务学习：
6.1 同时训练三个模型，使用共享层来提高信息共享和整体性能


DataPreprocessing类：
__init__(self, frames_dir: str, labels_dir: str): 构造函数, 初始化frames_dir和labels_dir
load_data(self): 从frames文件夹的每个子文件夹中加载每个视频的帧， 从labels文件夹中加载每个视频帧的标签，将每个视频的帧列表预期对应的标签作为一个元组储存在data列表中
resize_frame(self, frame: np.ndarray): 调整帧的大小，将帧调整为256*256的大小
split_data(self, train_ratio: float, val_ratio: float, test_ratio: float): 划分数据集, 将数据集划分为训练集，验证集和测试集
save_data(self, save_dir: str): 保存预处理数据集, 将数据集保存为pickle文件

Image2Sketch类：
__init__(self, data_path: str): 构造函数，加载数据集
load_data(self): 加载预处理后的数据集
preprocess_label(self, labels, image_shape): 根据关键点生成草图图像；labels(list)关键点列别，image_shape(tuple):图像的形状，return:草图图像 np.array
build_generator(self): 构建生成器模型
build_discriminator(self): 构建多尺度判别器模型
build_cgan(self): 构建cGAN模型，将生成器和判别器连接起来
train(self, epochs, batch_size): 训练cGAN模型
evaluate(self): 在验证集上评估模型性能
test(self): 在测试集上评估模型性能，生成草图
save(self, save_dir: str): 保存训练好的模型

MotionPrediction类：
__init__(self, data_path: str): 构造函数，加载数据集
load_data(self): 加载预处理后的数据集
preprocess_label(self, labels, image_shape): 根据关键点生成草图图像；labels(list)关键点列别，image_shape(tuple):图像的形状，return:草图图像 np.array
build_model(self, input_shape, filters = 64, kernel_size = (3, 3), activation = 'tanh'): 构建Seq2Seq模型，使用ConvLSTM做编码器和解码器
train(self, epochs, batch_size): 训练ConvLSTM模型
evaluate(self): 在验证集上评估模型性能
test(self): 在测试集上评估模型性能，生成未来帧的草图
save(self, save_dir: str): 保存训练好的模型
load_pretrained_model(self, encoder_model):将与训练的编码器模型参数加载到Seq2Seq的编码器部分

Sketch2Image类：
__init__(self, data_path: str): 构造函数，加载数据集
load_data(self): 加载预处理后的数据集
build_generator(self): 构建生成器模型
build_discriminator(self, input_shape, filters = 64, kernel_size = (3, 3)): 构建多尺度判别器模型
build_cgan(self): 构建cGAN模型，将生成器和判别器连接起来
train(self, epochs, batch_size): 训练cGAN模型
evaluate(self): 在验证集上评估模型性能
test(self): 在测试集上评估模型性能，生成未来帧的图像
save(self, save_dir: str): 保存训练好的模型
load_pretrained_model(self, generator_model, discriminator_model): 将与训练的生成器和判别器模型参数加载到cGAN的生成器和判别器部分


多任务学习和阶段式训练：
三个模型共享一部分卷积层
shared_conv_layer = Conv2D(filters, kernel_size(3, 3), padding = 'same', activation = 'relu')
def build_image2sketch_generator(input_shape):
    inputs = Input(shape = input_shape)
    x = shared_conv_layer(inputs)
    


Training Process:
1.阶段训练：
1.1 训练cGAN
1.1.1 加载数据
1.1.2 初始化并构建cGAN模型
1.1.3 训练cGAN模型,评估后保存模型
1.2 使用cGAN的输出训练Seq2Seq模型
1.2.1 使用cGAN生成的草图作为输入
1.2.2 初始化并构建Seq2Seq模型
1.2.3 训练Seq2Seq模型,评估后保存模型
1.3 训练cGAN
1.3.1 使用对应的frames的数据集以及Seq2Seq生成的草图作为输入
1.3.2 初始化并构建cGAN模型
1.3.3 训练cGAN模型,评估后保存模型

2. 多任务学习
同时训练三个模型，使用共享层来提高信息共享和整体性能

3. 迁移学习：使用cGAN(Image)


'''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import DataPreprocessing
from models import Image2Sketch, MotionPrediction, Sketch2Image, SharedConvLayers, MultiTaskLearning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备

def main():
    print("开始数据预处理...")
    # 数据预处理
    frames_dir = '/content/Penn_Action/frames'
    labels_dir = '/content/Penn_Action/labels'
    save_dir_of_pickle = '/content/data/data.pickle'
    sketch_dir = '/content/data/sketches'

    data_preprocessing = DataPreprocessing(frames_dir, labels_dir)
    data_preprocessing.load_data_from_frames_and_labels_folder_and_return_data_list()
    print("数据已加载。")
    
    data_preprocessing.resize_frame_two_five_six_square()
    print("数据已调整为256x256。")
    
    train_data, val_data, test_data = data_preprocessing.split_data(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    print("数据已拆分为训练、验证和测试集。")
    
    data_preprocessing.save_data_as_pickle_file(save_dir_of_pickle=save_dir_of_pickle)
    print("数据已保存为pickle文件。")

    # 初始化多任务学习模型
    print("初始化多任务学习模型...")

    print("初始化共享卷积层...")
    shared_layers = SharedConvLayers(input_channels=3).to(device)
    print("共享卷积层初始化完成。")

    print("初始化 Image2Sketch 模型...")
    image2sketch = Image2Sketch(save_dir_of_pickle=save_dir_of_pickle, shared_layers=shared_layers).to(device)
    print("Image2Sketch 模型初始化完成。")

    print("初始化 MotionPrediction 模型...")
    motion_prediction = MotionPrediction(labels_dir=labels_dir, sketch_dir=sketch_dir, shared_layers=shared_layers).to(device)
    print("MotionPrediction 模型初始化完成。")

    print("初始化 Sketch2Image 模型...")
    sketch2image = Sketch2Image(save_dir_of_pickle=save_dir_of_pickle, shared_layers=shared_layers).to(device)
    print("Sketch2Image 模型初始化完成。")

    print("初始化 MultiTaskLearning 模型...")
    multitask_model = MultiTaskLearning(shared_layers=shared_layers, image2sketch=image2sketch, motion_prediction=motion_prediction, sketch2image=sketch2image).to(device)
    print("MultiTaskLearning 模型初始化完成。")

    print("多任务学习模型初始化完成。")


    # 加载数据
    print("加载数据...")
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=32, shuffle=False)
    print("数据加载完成。")

    # 训练多任务学习模型
    print("开始训练模型...")
    train(multitask_model, train_loader, val_loader, epochs=100)

def train(model, train_loader, val_loader, epochs):
    # 定义损失函数和优化器
    criterion_gan = torch.nn.BCELoss()  # Binary Cross Entropy Loss
    criterion_pixelwise = torch.nn.L1Loss()  # L1 Loss
    criterion_seq2seq = torch.nn.HuberLoss()  # Huber Loss
    optimizer_g = optim.Adam(model.image2sketch.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.image2sketch.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_seq2seq = optim.Adam(model.motion_prediction.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_sketch2image_g = optim.Adam(model.sketch2image.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_sketch2image_d = optim.Adam(model.sketch2image.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        model.train()
        train_loss_g = 0.0  # 生成器损失
        train_loss_d = 0.0  # 判别器损失
        train_loss_seq2seq = 0.0  # Seq2Seq模型损失
        train_loss_sketch2image_g = 0.0  # Sketch2Image生成器损失
        train_loss_sketch2image_d = 0.0  # Sketch2Image判别器损失
        
        print(f"开始第 {epoch + 1} 轮训练...")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 处理cGAN模型
            shared_features = model.shared_layers(inputs)  # 共享卷积层
            output_g, output_d = model.image2sketch(shared_features)  # 生成器和判别器的输出

            real_labels = torch.ones((inputs.size(0), 1), device=device)  # 真实标签
            fake_labels = torch.zeros((inputs.size(0), 1), device=device)  # 假标签

            # 生成器损失
            loss_g_gan = criterion_gan(output_d, real_labels)
            loss_g_pixelwise = criterion_pixelwise(output_g, targets)
            loss_g = loss_g_gan + loss_g_pixelwise

            # 判别器损失
            loss_d_real = criterion_gan(output_d, real_labels)
            loss_d_fake = criterion_gan(output_d, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            # 反向传播和优化
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # 更新训练损失
            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()

            # 处理Seq2Seq模型
            shared_features = model.shared_layers(inputs)  # 共享卷积层
            output_seq2seq = model.motion_prediction(shared_features)

            # 计算损失
            loss_seq2seq = criterion_seq2seq(output_seq2seq, targets)

            # 反向传播和优化
            optimizer_seq2seq.zero_grad()
            loss_seq2seq.backward()
            optimizer_seq2seq.step()

            # 更新训练损失
            train_loss_seq2seq += loss_seq2seq.item()

            # 处理Sketch2Image模型(使用迁移学习)
            shared_features = model.shared_layers(inputs)  # 共享卷积层
            outputs_sketch2image_g, outputs_sketch2image_d = model.sketch2image(shared_features)

            # 计算生成器损失
            loss_sketch2image_g_gan = criterion_gan(outputs_sketch2image_d, real_labels)
            loss_sketch2image_g_pixelwise = criterion_pixelwise(outputs_sketch2image_g, targets)
            loss_sketch2image_g = loss_sketch2image_g_gan + loss_sketch2image_g_pixelwise

            # 计算判别器损失
            loss_sketch2image_d_real = criterion_gan(outputs_sketch2image_d, real_labels)
            loss_sketch2image_d_fake = criterion_gan(outputs_sketch2image_d, fake_labels)
            loss_sketch2image_d = (loss_sketch2image_d_real + loss_sketch2image_d_fake) / 2

            # 反向传播和优化
            optimizer_sketch2image_g.zero_grad()
            loss_sketch2image_g.backward()
            optimizer_sketch2image_g.step()

            optimizer_sketch2image_d.zero_grad()
            loss_sketch2image_d.backward()
            optimizer_sketch2image_d.step()

            # 更新训练损失
            train_loss_sketch2image_g += loss_sketch2image_g.item()
            train_loss_sketch2image_d += loss_sketch2image_d.item()

        # 打印训练损失
        print(f'Epoch {epoch + 1}/{epochs}, G_Loss: {train_loss_g / len(train_loader):.4f}, D_Loss: {train_loss_d / len(train_loader):.4f}, Seq2Seq_Loss: {train_loss_seq2seq / len(train_loader):.4f}, Sketch2Image_G_Loss: {train_loss_sketch2image_g / len(train_loader):.4f}, Sketch2Image_D_Loss: {train_loss_sketch2image_d / len(train_loader):.4f}')

        # 验证模型
        model.eval()
        val_loss_g = 0.0
        val_loss_d = 0.0
        val_loss_seq2seq = 0.0
        val_loss_sketch2image_g = 0.0
        val_loss_sketch2image_d = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 处理cGAN模型
                shared_features = model.shared_layers(inputs)  # 共享卷积层
                output_g, output_d = model.image2sketch(shared_features)
                loss_g_gan = criterion_gan(output_d, real_labels)
                loss_g_pixelwise = criterion_pixelwise(output_g, targets)
                loss_g = loss_g_gan + loss_g_pixelwise * 100

                loss_d_real = criterion_gan(output_d, real_labels)
                loss_d_fake = criterion_gan(output_d, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                val_loss_g += loss_g.item()
                val_loss_d += loss_d.item()

                # 处理Seq2Seq模型
                output_seq2seq = model.motion_prediction(shared_features)
                loss_seq2seq = criterion_seq2seq(output_seq2seq, targets)
                val_loss_seq2seq += loss_seq2seq.item()

                # 处理Sketch2Image模型
                outputs_sketch2image_g, outputs_sketch2image_d = model.sketch2image(shared_features)
                loss_sketch2image_g_gan = criterion_gan(outputs_sketch2image_d, real_labels)
                loss_sketch2image_g_pixelwise = criterion_pixelwise(outputs_sketch2image_g, targets)
                loss_sketch2image_g = loss_sketch2image_g_gan + loss_sketch2image_g_pixelwise

                loss_sketch2image_d_real = criterion_gan(outputs_sketch2image_d, real_labels)
                loss_sketch2image_d_fake = criterion_gan(outputs_sketch2image_d, fake_labels)
                loss_sketch2image_d = (loss_sketch2image_d_real + loss_sketch2image_d_fake) / 2

                val_loss_sketch2image_g += loss_sketch2image_g.item()
                val_loss_sketch2image_d += loss_sketch2image_d.item()

        # 打印验证损失
        print(f'Epoch {epoch + 1}/{epochs}, Val_G_Loss: {val_loss_g / len(val_loader):.4f}, Val_D_Loss: {val_loss_d / len(val_loader):.4f}, Val_Seq2Seq_Loss: {val_loss_seq2seq / len(val_loader):.4f}, Val_Sketch2Image_G_Loss: {val_loss_sketch2image_g / len(val_loader):.4f}, Val_Sketch2Image_D_Loss: {val_loss_sketch2image_d / len(val_loader):.4f}')

if __name__ == '__main__':
    main()
