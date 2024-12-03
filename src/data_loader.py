import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_train_data(self):
        """加载训练数据"""
        # 使用pickle加载.pkl格式的训练数据
        with open(self.config.TRAIN_DATA_PATH, 'rb') as f:
            X = pickle.load(f)
        # 使用numpy加载.npy格式的标签
        y = np.load(self.config.TRAIN_LABELS_PATH)
        return X, y
    
    def load_test_data(self):
        """加载测试数据"""
        # 使用pickle加载.pkl格式的测试数据
        with open(self.config.TEST_DATA_PATH, 'rb') as f:
            X_test = pickle.load(f)
        return X_test
    
    def create_validation_split(self, X, y, val_size=0.2):
        """创建验证集"""
        return train_test_split(
            X, y,
            test_size=val_size,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
