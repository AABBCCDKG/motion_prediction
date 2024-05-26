'''
数据预处理
使用Penn_Actions数据集，在Penn_Action数据集中，有文件夹frames和labels，frames文件夹中包含2326个子文件夹，每个子文件夹包含一个视频的所有帧(从000001.jpg开始到结束)，labels文件夹中包含2326个.mat文件，每个.mat文件包含一个视频所有帧的标签。

1. 加载数据：
1.1 从frames文件夹中加载每个视频的帧，每个子文件夹（0001到2326）包含一个视频的所有帧，按照文件名的顺序（例如从000001.jpg到结束）读取帧，以确保视频帧的顺序正确。

1.2 从labels文件夹中加载每个视频帧的标签，每个.mat文件（0001.mat到2326.mat）包含一个视频所有帧的标签
1.3 将每个视频的帧列表预期对应的标签作为一个元组储存在data列表中

2. 调整帧的大小：
2.1 调整帧的大小，将帧调整为256*256的大小

3. 划分数据集：
3.1 按照给定的比例划分为训练集，验证集和测试集
3.2 使用train_test_split方法确保数据的随机划分

4. 保存预处理数据集：
4.1 将预处理数据集保存到指定的路径，便于后续使用
4.2 数据集保存为pickle文件，方便后续读取

5. 在测试模型时，将上传的视频拆解成视频帧

导入所需的库

Class:
DataPreprocessing

Attributes:
frames_dir(str): 储存帧的文件夹路径
labels_dir(str): 储存标签的文件夹路径
data(list): 储存帧和标签的列表
train_data(list): 训练数据集（帧和标签）
val_data(list): 验证数据集（帧和标签）
test_data(list): 测试数据集（帧和标签）

Methods:
def __init__(self, frames_dir: str, labels_dir: str): 构造函数, 初始化frames_dir和labels_dir
def load_data_from_frames_and_labels_folder_and_return_data_list(self): 从frames文件夹的每个子文件夹中加载每个视频的帧， 从labels文件夹中加载每个视频帧的标签，同时调用generate_sketch_from_keypoints将每个视频帧的标签从keypoints转换成草图，将每个视频的帧列表预期对应的标签（草图形式）作为一个元组储存在data列表中
def generate_sketch_from_keypoints(self, labels, frames):根据labels文件夹中的.mat文件内的关键点生成草图，依然是.mat文件，每个.mat文件包含一个视频所有帧的标签
def resize_frame_two_five_six_square(self): 调整帧的大小，将帧调整为256*256的大小
def split_data(self, train_ratio: float, val_ratio: float, test_ratio: float): 划分数据集, 将数据集划分为训练集，验证集和测试集
def save_data_as_pickle_file(self, save_dir_of_pickle: str): 保存预处理数据集, 将数据集保存为pickle文件
def extract_frames_when_uploading_video_in_testing(video_path, generated_frames_store_folder):将上传的视频拆解成视频帧

'''
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

# frames和labels文件夹路径
frames_dir = '/content/Penn_Action/frames'
labels_dir = '/content/Penn_Action/labels'

class DataPreprocessing:
    def __init__(self, frames_dir: str, labels_dir: str):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data_from_frames_and_labels_folder_and_return_data_list(self):
        print("开始加载数据...")
        for video_id in range(1, 50):
            video_frames_path = os.path.join(self.frames_dir, f'{video_id:04d}')
            video_labels_path = os.path.join(self.labels_dir, f'{video_id:04d}.mat')

            if not os.path.exists(video_frames_path) or not os.path.exists(video_labels_path):
                print(f"视频或标签文件缺失，跳过视频 {video_id}")
                continue

            frames = []
            for frame_name in sorted(os.listdir(video_frames_path)):
                frame_path = os.path.join(video_frames_path, frame_name)
                frame = cv2.imread(frame_path)
                frames.append(frame)

            labels = scipy.io.loadmat(video_labels_path)
            x_coords = labels['x']
            y_coords = labels['y']
            sketches = self.generate_sketch_from_keypoints(x_coords, y_coords, frames)
            self.data.append((frames, sketches))

            if video_id % 100 == 0:
                print(f"已加载 {video_id} 个视频")

        # 打印加载的数据长度以进行调试
        print(f"Total videos loaded: {len(self.data)}")

    def generate_sketch_from_keypoints(self, x_coords, y_coords, frames):
        sketches = []
        for x, y, frame in zip(x_coords.T, y_coords.T, frames):
            sketch = np.zeros(frame.shape[:2], dtype=np.uint8)
            for i in range(len(x) - 1):
                x1, y1 = x[i], y[i]
                x2, y2 = x[i + 1], y[i + 1]
                cv2.line(sketch, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
            sketches.append(sketch)
        return sketches

    def resize_frame_two_five_six_square(self):
        print("开始调整帧大小为256x256...")
        for i, (frames, sketches) in enumerate(self.data):
            resized_frames = [cv2.resize(frame, (256, 256)) for frame in frames]
            resized_sketches = [cv2.resize(sketch, (256, 256)) for sketch in sketches]
            self.data[i] = (resized_frames, resized_sketches)
            if i % 100 == 0:
                print(f"已处理 {i} 个视频")
        print("帧大小调整完成。")

   
    def split_data(self, train_ratio: float, val_ratio: float, test_ratio: float):
        print("开始拆分数据...")
        try:
            train_val_data, test_data = train_test_split(self.data, test_size=test_ratio)
            print(f"训练+验证集: {len(train_val_data)}, 测试集: {len(test_data)}")

            train_data, val_data = train_test_split(train_val_data, test_size=val_ratio / (train_ratio + val_ratio))
            print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

            # 找出所有样本的最大帧数
            max_frame_len = max(
                [len(item[0]) for item in train_data + val_data + test_data]
            )

            # 定义一个填充序列的函数
            def pad_sequence(seq, max_len, frame_shape):
                padded = np.zeros((max_len, *frame_shape), dtype=seq.dtype)
                padded[:len(seq)] = seq
                return padded

            # 获取每个视频帧的形状 (H, W, C)
            frame_shape = train_data[0][0][0].shape

            # 处理训练数据
            print("转换训练数据为张量...")
            train_inputs = []
            train_targets = []
            for i, (frames, sketches) in enumerate(train_data):
                try:
                    padded_frames = pad_sequence(np.array(frames), max_frame_len, frame_shape)
                    padded_sketches = pad_sequence(np.array(sketches), max_frame_len, (256, 256))
                    for j in range(1, len(frames)):
                        if j >= len(padded_frames) or j >= len(padded_sketches):
                            continue
                        input_tensor = np.concatenate((padded_frames[j-1], padded_frames[j]), axis=2)
                        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1).to(self.device)  # 匹配 PyTorch 的 (C, H, W) 格式，并移动到 GPU
                        sketch_tensor = torch.tensor(padded_sketches[j], dtype=torch.float32).unsqueeze(0).to(self.device)  # 添加通道维度，并移动到 GPU
                        train_inputs.append(input_tensor)
                        train_targets.append(sketch_tensor)
                    if i % 10 == 0:
                        print(f"已转换 {i} 个训练数据")
                except Exception as e:
                    print(f"在转换训练数据索引 {i} 时发生错误: {e}")

            print("训练数据转换完成。")

            # 处理验证数据
            print("转换验证数据为张量...")
            val_inputs = []
            val_targets = []
            for i, (frames, sketches) in enumerate(val_data):
                try:
                    padded_frames = pad_sequence(np.array(frames), max_frame_len, frame_shape)
                    padded_sketches = pad_sequence(np.array(sketches), max_frame_len, (256, 256))
                    for j in range(1, len(frames)):
                        if j >= len(padded_frames) or j >= len(padded_sketches):
                            continue
                        input_tensor = np.concatenate((padded_frames[j-1], padded_frames[j]), axis=2)
                        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1).to(self.device)  # 匹配 PyTorch 的 (C, H, W) 格式，并移动到 GPU
                        sketch_tensor = torch.tensor(padded_sketches[j], dtype=torch.float32).unsqueeze(0).to(self.device)  # 添加通道维度，并移动到 GPU
                        val_inputs.append(input_tensor)
                        val_targets.append(sketch_tensor)
                    if i % 10 == 0:
                        print(f"已转换 {i} 个验证数据")
                except Exception as e:
                    print(f"在转换验证数据索引 {i} 时发生错误: {e}")

            print("验证数据转换完成。")

            # 处理测试数据
            print("转换测试数据为张量...")
            test_inputs = []
            test_targets = []
            for i, (frames, sketches) in enumerate(test_data):
                try:
                    padded_frames = pad_sequence(np.array(frames), max_frame_len, frame_shape)
                    padded_sketches = pad_sequence(np.array(sketches), max_frame_len, (256, 256))
                    for j in range(1, len(frames)):
                        if j >= len(padded_frames) or j >= len(padded_sketches):
                            continue
                        input_tensor = np.concatenate((padded_frames[j-1], padded_frames[j]), axis=2)
                        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1).to(self.device)  # 匹配 PyTorch 的 (C, H, W) 格式，并移动到 GPU
                        sketch_tensor = torch.tensor(padded_sketches[j], dtype=torch.float32).unsqueeze(0).to(self.device)  # 添加通道维度，并移动到 GPU
                        test_inputs.append(input_tensor)
                        test_targets.append(sketch_tensor)
                    if i % 10 == 0:
                        print(f"已转换 {i} 个测试数据")
                except Exception as e:
                    print(f"在转换测试数据索引 {i} 时发生错误: {e}")

            print("测试数据转换完成。")

            self.train_data = (torch.stack(train_inputs), torch.stack(train_targets))
            self.val_data = (torch.stack(val_inputs), torch.stack(val_targets))
            self.test_data = (torch.stack(test_inputs), torch.stack(test_targets))

            print(f"数据拆分完成。训练集: {len(train_data)} 验证集: {len(val_data)} 测试集: {len(test_data)}")
            return self.train_data, self.val_data, self.test_data

        except Exception as e:
            print(f"在拆分数据时发生错误: {e}")


    
    def save_data_as_pickle_file(self, save_dir_of_pickle: str):
        print("开始保存数据为pickle文件...")
        os.makedirs(save_dir_of_pickle, exist_ok=True)
        
        print("保存训练数据...")
        with open(os.path.join(save_dir_of_pickle, 'train_data.pkl'), 'wb') as f:
            pickle.dump(self.train_data, f)
        print("训练数据保存完成。")
        
        print("保存验证数据...")
        with open(os.path.join(save_dir_of_pickle, 'val_data.pkl'), 'wb') as f:
            pickle.dump(self.val_data, f)
        print("验证数据保存完成。")
        
        print("保存测试数据...")
        with open(os.path.join(save_dir_of_pickle, 'test_data.pkl'), 'wb') as f:
            pickle.dump(self.test_data, f)
        print("测试数据保存完成。")
        
        print("数据保存完成。")

    def extract_frames_when_uploading_video_in_testing(self, video_path, generated_frames_store_folder):
        cap = cv2.VideoCapture(video_path)  # 读取视频
        frame_id = 0  # 帧的id
        os.makedirs(generated_frames_store_folder, exist_ok=True)  # 创建文件夹
        while cap.isOpened():  # 当视频打开时
            ret, frame = cap.read()  # 读取视频的帧
            if not ret:  # 如果没有读取到帧
                break  # 退出循环
            frame_path = os.path.join(generated_frames_store_folder, f'{frame_id:06d}.jpg')  # 生成帧的路径
            cv2.imwrite(frame_path, frame)  # 保存帧
            frame_id += 1
        cap.release()

# Example usage:
if __name__ == "__main__":
    save_dir_of_pickle = '/content/processed_data'

    data_preprocessing = DataPreprocessing(frames_dir, labels_dir)
    data_preprocessing.load_data_from_frames_and_labels_folder_and_return_data_list()
    data_preprocessing.resize_frame_two_five_six_square()
    data_preprocessing.split_data(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    data_preprocessing.save_data_as_pickle_file(save_dir_of_pickle)

