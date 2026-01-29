"""
数据加载模块
支持CIFAR-10的联邦划分和类增量任务
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import os


class CIFAR10Federated:
    """
    CIFAR-10 联邦类增量学习数据集
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        num_clients: int = 5,
        alpha: float = 0.5,
        seed: int = 42
    ):
        self.data_root = data_root
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet分布参数
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # 加载完整数据集
        self.train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=self.test_transform
        )
        
        # 类别名称
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # 按类别索引数据
        self.train_indices_by_class = self._get_indices_by_class(self.train_dataset)
        self.test_indices_by_class = self._get_indices_by_class(self.test_dataset)
        
        # 客户端数据划分（使用Dirichlet分布模拟Non-IID）
        self.client_indices = None  # 延迟到具体任务时划分
    
    def _get_indices_by_class(self, dataset) -> Dict[int, List[int]]:
        """按类别获取样本索引"""
        indices_by_class = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            indices_by_class[label].append(idx)
        return indices_by_class
    
    def partition_data_dirichlet(
        self,
        class_indices: Dict[int, List[int]],
        classes: List[int]
    ) -> Dict[int, List[int]]:
        """
        使用Dirichlet分布划分数据到各客户端
        
        Args:
            class_indices: 每个类别的样本索引
            classes: 当前任务包含的类别
        
        Returns:
            client_indices: 每个客户端的样本索引
        """
        client_indices = {k: [] for k in range(self.num_clients)}
        
        for cls in classes:
            indices = np.array(class_indices[cls])
            np.random.shuffle(indices)
            
            # Dirichlet分布生成比例
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # 按比例分配
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] = len(indices) - proportions[:-1].sum()  # 确保总和正确
            
            start = 0
            for k in range(self.num_clients):
                end = start + proportions[k]
                client_indices[k].extend(indices[start:end].tolist())
                start = end
        
        return client_indices
    
    def get_task_data(
        self,
        task_classes: List[int],
        client_id: Optional[int] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        获取特定任务的数据
        
        Args:
            task_classes: 任务包含的类别ID列表
            client_id: 如果指定，返回该客户端的数据；否则返回全部数据
        
        Returns:
            train_dataset, test_dataset
        """
        # 划分当前任务的数据
        task_train_indices = self.partition_data_dirichlet(
            self.train_indices_by_class, task_classes
        )
        
        if client_id is not None:
            train_indices = task_train_indices[client_id]
        else:
            train_indices = []
            for k in range(self.num_clients):
                train_indices.extend(task_train_indices[k])
        
        # 测试集：包含当前任务所有类别
        test_indices = []
        for cls in task_classes:
            test_indices.extend(self.test_indices_by_class[cls])
        
        train_subset = Subset(self.train_dataset, train_indices)
        test_subset = Subset(self.test_dataset, test_indices)
        
        return train_subset, test_subset
    
    def get_client_task_data(
        self,
        task_classes: List[int]
    ) -> Dict[int, Tuple[List[int], List[int]]]:
        """
        获取所有客户端在特定任务上的数据划分
        
        Returns:
            {client_id: (train_indices, local_classes)}
        """
        # 划分数据
        client_indices = self.partition_data_dirichlet(
            self.train_indices_by_class, task_classes
        )
        
        client_data = {}
        for k in range(self.num_clients):
            indices = client_indices[k]
            # 获取该客户端实际拥有的类别
            local_classes = set()
            for idx in indices:
                _, label = self.train_dataset[idx]
                if label in task_classes:
                    local_classes.add(label)
            client_data[k] = (indices, list(local_classes))
        
        return client_data
    
    def get_cumulative_test_data(
        self,
        classes_so_far: List[int]
    ) -> Dataset:
        """
        获取到目前为止所有类别的测试数据
        """
        test_indices = []
        for cls in classes_so_far:
            test_indices.extend(self.test_indices_by_class[cls])
        return Subset(self.test_dataset, test_indices)


class TaskDataset(Dataset):
    """
    任务数据集包装器
    支持从原始数据集中提取特定类别的数据
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        indices: List[int],
        class_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            base_dataset: 基础数据集
            indices: 使用的样本索引
            class_mapping: 原始类别到新类别的映射（可选）
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_mapping = class_mapping
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        
        if self.class_mapping is not None:
            label = self.class_mapping.get(label, label)
        
        return image, label


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# 测试代码
if __name__ == "__main__":
    # 创建联邦数据集
    fed_data = CIFAR10Federated(
        data_root="./data",
        num_clients=5,
        alpha=0.5
    )
    
    # 定义任务
    tasks = [
        [0, 1, 2, 3],  # airplane, automobile, bird, cat
        [4, 5, 6, 7],  # deer, dog, frog, horse
        [8, 9]         # ship, truck
    ]
    
    print("=" * 50)
    print("CIFAR-10 Federated Class-Incremental Learning")
    print("=" * 50)
    
    for t, task_classes in enumerate(tasks):
        print(f"\nTask {t + 1}: {[fed_data.class_names[c] for c in task_classes]}")
        
        client_data = fed_data.get_client_task_data(task_classes)
        
        for k, (indices, local_classes) in client_data.items():
            class_names = [fed_data.class_names[c] for c in local_classes]
            print(f"  Client {k}: {len(indices)} samples, classes: {class_names}")
