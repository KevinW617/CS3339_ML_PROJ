from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.svd = TruncatedSVD(
            n_components=config.SVD_COMPONENTS,
            random_state=config.RANDOM_STATE
        )
        self.scaler = StandardScaler()
        
    def fit_transform(self, X):
        """对训练数据进行拟合和转换"""
        X_reduced = self.svd.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_reduced)
        
        # 计算并打印方差解释率
        explained_variance_ratio = self.svd.explained_variance_ratio_.sum()
        print(f"解释方差比率: {explained_variance_ratio:.4f}")
        
        return X_scaled
    
    def transform(self, X):
        """对新数据进行转换"""
        X_reduced = self.svd.transform(X)
        X_scaled = self.scaler.transform(X_reduced)
        return X_scaled