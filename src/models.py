from sklearn.linear_model import SGDClassifier, LogisticRegression
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DeepSparseClassifier(nn.Module):
    def __init__(self, input_dim=500, hidden_dims=[1024, 512, 256], num_classes=20, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层：大规模特征提取
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dims[0])])
        
        # 中间层：残差块
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]))
            
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 残差连接的1x1卷积
        self.shortcuts = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.shortcuts.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
    def forward(self, x):
        # 第一层
        x = self.layers[0](x)
        x = self.batch_norms[0](x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        prev = x
        
        # 中间层（带残差连接）
        for i in range(1, len(self.layers)):
            identity = self.shortcuts[i-1](prev)
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            x = x + identity  # 残差连接
            prev = x
            
        # 输出层
        x = self.output_layer(x)
        return x

class DeepLearningWrapper:
    def __init__(self, input_dim=500, hidden_dims=[1024, 512, 256], num_classes=20,
                 lr=0.001, batch_size=128, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = DeepSparseClassifier(input_dim, hidden_dims, num_classes)
        self.device = device
        self.model.to(device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', 
                                                                   factor=0.5, patience=5, verbose=True)
        
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_acc = 0
        patience = 10
        no_improve = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{self.epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%')
            
            # 早停
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    break
            
            self.scheduler.step(acc)
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = []
            for i in range(0, len(X), self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                batch_output = self.model(batch_x)
                outputs.append(batch_output.cpu())
            outputs = torch.cat(outputs, dim=0)
            _, predicted = outputs.max(1)
        return predicted.numpy()
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class ModelFactory:
    def __init__(self, config):
        self.config = config
        
    def get_linear_svm(self):
        return SGDClassifier(
            **self.config.SVM_PARAMS,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
    
    def get_lightgbm(self):
        return lgb.LGBMClassifier(
            **self.config.LGBM_PARAMS,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
    
    def get_logistic_regression(self):
        return LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            C=1.0,
            multi_class='multinomial',
            max_iter=200,
            n_jobs=-1,
            random_state=self.config.RANDOM_STATE,
            verbose=1
        )
    
    def get_deep_learning(self):
        return DeepLearningWrapper(
            input_dim=self.config.SVD_COMPONENTS,
            hidden_dims=[1024, 512, 256],
            num_classes=20,
            lr=0.001,
            batch_size=128,
            epochs=100
        )
