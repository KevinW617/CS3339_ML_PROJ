from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.models import ModelFactory
from src.evaluator import ModelEvaluator
import pandas as pd

def save_predictions(predictions, model_name):
    """保存每个模型的预测结果"""
    submission_df = pd.DataFrame({
        'Id': range(len(predictions)),
        'Category': predictions
    })
    submission_path = f'./data/submission_{model_name.lower().replace(" ", "_")}.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")

def main():
    # 初始化配置
    config = Config()
    
    # 加载数据
    print("Loading data...")
    data_loader = DataLoader(config)
    X, y = data_loader.load_train_data()
    X_test = data_loader.load_test_data()
    X_train, X_val, y_train, y_val = data_loader.create_validation_split(X, y)
    
    # 预处理
    print("\nPreprocessing data...")
    preprocessor = Preprocessor(config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # 初始化模型工厂和评估器
    model_factory = ModelFactory(config)
    evaluator = ModelEvaluator(config)
    
    # 评估所有模型并生成预测
    models = {
        'Linear_SVM': model_factory.get_linear_svm(),
        'LightGBM': model_factory.get_lightgbm(),
        'Logistic_Regression': model_factory.get_logistic_regression(),
        'Deep_Learning': model_factory.get_deep_learning()
    }
    
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")
        try:
            # 训练模型
            model.fit(X_train_processed, y_train)
            
            # 验证集评估
            val_score = model.score(X_val_processed, y_val)
            print(f"Validation accuracy for {name}: {val_score:.4f}")
            
            # 生成测试集预测
            print(f"Generating predictions for {name}...")
            test_predictions = model.predict(X_test_processed)
            
            # 保存预测结果
            save_predictions(test_predictions, name)
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()