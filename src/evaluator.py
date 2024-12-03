from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        
    def cross_validate(self, model, X, y):
        """执行交叉验证"""
        skf = StratifiedKFold(
            n_splits=self.config.N_SPLITS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            scores.append(score)
            
            print(f"Fold {fold} Accuracy: {score:.4f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\nMean Accuracy: {mean_score:.4f} (±{std_score:.4f})")
        
        return mean_score, std_score
    
    def evaluate_final_model(self, model, X_train, y_train, X_val, y_val):
        """评估最终模型"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        print("\nValidation Set Performance:")
        print(classification_report(y_val, y_pred))
        
        return model