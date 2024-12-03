class Config:
    # 数据相关
    TRAIN_DATA_PATH = "./data/train_feature.pkl"
    TRAIN_LABELS_PATH = "./data/train_labels.npy"
    TEST_DATA_PATH = "./data/test_feature.pkl"
    
    # 预处理参数
    SVD_COMPONENTS = 500
    RANDOM_STATE = 42
    
    # 模型参数
    # SVM
    SVM_PARAMS = {
        'loss': 'hinge',
        'penalty': 'l2',
        'alpha': 1e-4,
        'max_iter': 1000,
        'tol': 1e-3
    }
    
    # LightGBM
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 20,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    # Logistic Regression
    LR_PARAMS = {
        'penalty': 'l1',
        'solver': 'saga',
        'C': 1.0,
        'multi_class': 'multinomial',
        'max_iter': 1000
    }
    
    # 交叉验证
    N_SPLITS = 5