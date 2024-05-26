'''
test.py 

1. 导入数据处理，模型构建，模型训练，模型评估的所需要的库

2. 数据预处理：
2.1 使用DataPreprocessing类加载预处理后的测试集数据
2.2 初始化DataPreprocessing类
2.2.1 设置帧和标签的文件夹路径
2.2.2 调用load_data()方法加载数据

3 加载模型：
3.1 使用Image2Sketch类加载训练好的生成器模型
3.2 使用MotionPreditcion类加载训练好的ConvLSTM模型
3.3 使用Sketch2Image类加载训练好的cGAN模型 

4. 测试模型：
4.1 使用加载的Image2Sketch生成器模型对测试集进行草图生成
4.2 使用加载的MotionPrediction模型对测试集进行未来帧草图预测
4.3 使用加载的Sketch2Image cGAN模型对测试集进行未来帧图像生成

5. 评估结果：
5.1 计算生成草图的和真实草图之间的相似度
5.2 计算未来帧草图预测的和真实未来帧草图之间的相似度
5.3 计算生成图像的和真实图像之间的相似度

6. 可视化结果：
6.1 显示和保存生成的草图，预测的草图和生成的图片
6.2 使用matplotlib显示生成的草图，预测的草图和生成的图片


DataPreprocessing:
__init__(self, frames_dir: str, labels_dir: str): 构造函数, 初始化frames_dir和labels_dir
load_data(self): 从frames文件夹的每个子文件夹中加载每个视频的帧， 从labels文件夹中加载每个视频帧的标签，将每个视频的帧列表预期对应的标签作为一个元组储存在data列表中

Image2Sketch:
__init__(self, data_path: str): 构造函数，加载数据集
load_model(self, model_path: str): 加载训练好的生成器模型
generate_sketch(self, images): 生成草图

MotionPrediction:
__init__(self, data_path: str): 构造函数，加载数据集
load_model(self, model_path: str): 加载训练好的ConvLSTM模型
predict(self, data): 预测未来帧草图

Sketch2Image:
__init__(self, data_path: str): 构造函数，加载数据集
load_model(self, model_path: str): 加载训练好的cGAN模型
generate_image(self, sketch, label): 生成未来帧图像

def visualize_results(original_images, generated_sketches, predicted_sketches, generated_images): 可视化结果

'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from data_preprocessing import DataPreprocessing
from models import Image2Sketch, MotionPrediction, Sketch2Image, SharedConvLayers, MultiTaskLearning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置设备

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型到指定设备
    model.to(device)
    model.eval()
    return model

def save_frames_as_video(frames, filename, fps=10):
    height, width = frames[0].shape[1], frames[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames: 
        frame = frame.squeeze().cpu().numpy() # 将数据从 GPU 移动到 CPU，并转换为 numpy 数组
        frame = (frame * 255).astype(np.uint8) # 将浮点数数据转换为 0-255 的整数
        if len(frame.shape) == 2: # 如果是单通道图像
            frame = cv2.merge([frame, frame, frame]) # 转换为三通道图像
        out.write(frame) # 写入视频

    out.release()

def test_model(multitask_model, test_loader, output_dir, batch_size):
    multitask_model.eval() # 设置为评估模式
    criterion_gan = torch.nn.BCELoss() # 定义损失函数
    criterion_pixelwise = torch.nn.L1Loss() # 定义损失函数
    criterion_seq2seq = torch.nn.HuberLoss() # 定义损失函数

    test_loss_g = 0.0
    test_loss_d = 0.0
    test_loss_seq2seq = 0.0
    test_loss_sketch2image_g = 0.0
    test_loss_sketch2image_d = 0.0

    real_labels = torch.ones(batch_size, 1).to(device) # 定义真实标签
    fake_labels = torch.zeros(batch_size, 1).to(device) # 定义假标签

    with torch.no_grad(): # 不进行梯度计算
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Process Image2Sketch model
            shared_features = multitask_model.shared_layers(inputs) # 提取共享特征
            output_g, output_d = multitask_model.image2sketch(shared_features) # 生成草图

            loss_g_gan = criterion_gan(output_d, real_labels) # 计算生成器损失
            loss_g_pixelwise = criterion_pixelwise(output_g, targets)
            loss_g = loss_g_gan + loss_g_pixelwise

            loss_d_real = criterion_gan(output_d, real_labels) # 计算判别器损失
            loss_d_fake = criterion_gan(output_d, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            test_loss_g += loss_g.item()
            test_loss_d += loss_d.item()

            # Process MotionPrediction model
            outputs_seq2seq = multitask_model.motion_prediction(shared_features)
            loss_seq2seq = criterion_seq2seq(outputs_seq2seq, targets)
            test_loss_seq2seq += loss_seq2seq.item()

            # Process Sketch2Image model
            outputs_sketch2image_g, output_sketch2image_d = multitask_model.sketch2image(shared_features)
            loss_sketch2image_g_gan = criterion_gan(output_sketch2image_d, real_labels)
            loss_sketch2image_g_pixelwise = criterion_pixelwise(outputs_sketch2image_g, targets)
            loss_sketch2image_g = loss_sketch2image_g_gan + loss_sketch2image_g_pixelwise

            loss_sketch2image_d_real = criterion_gan(output_sketch2image_d, real_labels)
            loss_sketch2image_d_fake = criterion_gan(output_sketch2image_d, fake_labels)
            loss_sketch2image_d = (loss_sketch2image_d_real + loss_sketch2image_d_fake) / 2

            test_loss_sketch2image_g += loss_sketch2image_g.item()
            test_loss_sketch2image_d += loss_sketch2image_d.item()

            # Save frames as video
            frames = outputs_sketch2image_g
            video_filename = f'{output_dir}/video_{i}.avi'
            save_frames_as_video(frames, video_filename)
    
    # 打印测试结果
    print(f'Test_G_Loss: {test_loss_g / len(test_loader):.4f}, Test_D_Loss: {test_loss_d / len(test_loader):.4f}, Test_Seq2Seq_Loss: {test_loss_seq2seq / len(test_loader):.4f}, Test_Sketch2Image_G_Loss: {test_loss_sketch2image_g / len(test_loader):.4f}, Test_Sketch2Image_D_Loss: {test_loss_sketch2image_d / len(test_loader):.4f}')

def main():
    # Data preprocessing
    frames_dir = 'data/frames'
    labels_dir = 'data/labels'
    data_preprocessing = DataPreprocessing(frames_dir, labels_dir)
    data_preprocessing.load_data_from_frames_and_labels_folder_and_return_data_list()
    data_preprocessing.resize_frame_two_five_six_square()
    train_data, val_data, test_data = data_preprocessing.split_data(0.7, 0.2, 0.1)
    data_preprocessing.save_data_as_pickle_file('data/data.pickle')

    # Initialize MultiTask Learning Model
    shared_layers = SharedConvLayers(input_channels=3)
    image2sketch = Image2Sketch(save_dir_of_pickle='data', shared_layers=shared_layers)
    motion_prediction = MotionPrediction(labels_dir='data/labels', sketch_dir='data/sketch', shared_layers=shared_layers)
    sketch2image = Sketch2Image(save_dir_of_pickle='data', shared_layers=shared_layers)
    multitask_model = MultiTaskLearning(shared_layers=shared_layers, image2sketch=image2sketch, motion_prediction=motion_prediction, sketch2image=sketch2image)

    # Load trained models
    multitask_model.image2sketch = load_model(multitask_model.image2sketch, 'path/to/saved_image2sketch_model.pth')
    multitask_model.motion_prediction = load_model(multitask_model.motion_prediction, 'path/to/saved_motion_prediction_model.pth')
    multitask_model.sketch2image = load_model(multitask_model.sketch2image, 'path/to/saved_sketch2image_model.pth')

    # Load test data
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=32, shuffle=False)

    # Test the MultiTask Learning Model
    output_dir = 'path/to/output_videos'
    test_model(multitask_model, test_loader, output_dir, batch_size=32)

if __name__ == '__main__':
    main()
