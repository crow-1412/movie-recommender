import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging

class HybridRecommender:
    def __init__(self, data_path, embedding_dim=50, lstm_units=32):
        """
        初始化混合推荐系统
        
        Args:
            data_path (str): ML-100K数据集路径
            embedding_dim (int): 嵌入维度
            lstm_units (int): LSTM单元数量
        """
        self.setup_logging()
        
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.seq_length = 5
        
        # 模型组件
        self.dnn_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        # 数据相关属性
        self.n_users = None
        self.n_movies = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        
        # 加载并预处理数据
        self.load_data()
        
    def setup_logging(self):
        """设置日志记录"""
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'experiment_log_{timestamp}.log'
        
        # 配置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 获取logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 清除之前的处理器
        self.logger.handlers = []
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 记录初始信息
        self.logger.info("=== 开始新的实验 ===")
        
    def load_data(self):
        """加载并预处理ML-100K数据集"""
        self.logger.info("加载数据...")
        
        # 读取评分数据
        ratings_file = os.path.join(self.data_path, 'u.data')
        self.ratings_df = pd.read_csv(ratings_file, sep='\t', 
                                    names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 获取用户和电影数量
        self.n_users = self.ratings_df['user_id'].max()
        self.n_movies = self.ratings_df['item_id'].max()
        
        self.logger.info(f"数据集包含 {self.n_users} 个用户和 {self.n_movies} 个电影")
        
        # 创建用户-项目矩阵
        self.create_user_item_matrix()
        
    def create_user_item_matrix(self):
        """创建用户-项目矩阵并计算相似度"""
        self.logger.info("创建用户-项目矩阵...")
        self.user_item_matrix = pd.pivot_table(
            self.ratings_df,
            values='rating',
            index='user_id',
            columns='item_id',
            fill_value=0
        )
        
        self.logger.info("计算项目相似度矩阵...")
        self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
    def build_models(self):
        """构建DNN和LSTM模型"""
        self.logger.info("构建DNN模型...")
        # 用户输入
        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.n_users + 1, self.embedding_dim)(user_input)
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        
        # 电影输入
        movie_input = Input(shape=(1,))
        movie_embedding = Embedding(self.n_movies + 1, self.embedding_dim)(movie_input)
        movie_vec = tf.keras.layers.Flatten()(movie_embedding)
        
        # 合并嵌入
        concat = Concatenate()([user_vec, movie_vec])
        
        # DNN层
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        
        # 输出层
        output = Dense(1, activation='sigmoid')(dense3)
        
        # 构建DNN模型
        self.dnn_model = Model(inputs=[user_input, movie_input], outputs=output)
        self.dnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.logger.info("构建LSTM模型...")
        # LSTM模型
        sequence_input = Input(shape=(self.seq_length, self.embedding_dim * 2))
        lstm1 = LSTM(self.lstm_units * 2, return_sequences=True)(sequence_input)
        dropout1 = Dropout(0.4)(lstm1)
        lstm2 = LSTM(self.lstm_units * 2, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.3)(lstm2)
        lstm3 = LSTM(self.lstm_units)(dropout2)
        dropout3 = Dropout(0.2)(lstm3)
        
        # 深度层
        dense1 = Dense(256, activation='relu')(dropout3)
        dropout4 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dropout4)
        dense3 = Dense(64, activation='relu')(dense2)
        
        # 输出层
        output = Dense(self.n_movies + 1, activation='softmax')(dense3)
        
        # 构建LSTM模型
        self.lstm_model = Model(inputs=sequence_input, outputs=output)
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def prepare_training_data(self):
        """准备训练数据"""
        self.logger.info("准备训练数据...")
        
        # 为DNN准备数据
        X_dnn = [
            self.ratings_df['user_id'].values,
            self.ratings_df['item_id'].values
        ]
        y_dnn = self.scaler.fit_transform(
            self.ratings_df['rating'].values.reshape(-1, 1)
        ).flatten()
        
        # 为LSTM准备数据
        sequences = []
        next_items = []
        
        # 按用户和时间戳排序
        sorted_ratings = self.ratings_df.sort_values(['user_id', 'timestamp'])
        
        for user_id in tqdm(sorted_ratings['user_id'].unique()):
            user_ratings = sorted_ratings[sorted_ratings['user_id'] == user_id]
            
            if len(user_ratings) <= self.seq_length:
                continue
                
            for i in range(len(user_ratings) - self.seq_length):
                seq = user_ratings.iloc[i:i+self.seq_length]
                next_item = user_ratings.iloc[i+self.seq_length]['item_id']
                
                # 获取序列特征
                seq_features = np.zeros((self.seq_length, self.embedding_dim * 2))
                sequences.append(seq_features)
                
                # 创建one-hot编码
                next_item_one_hot = np.zeros(self.n_movies + 1)
                next_item_one_hot[next_item] = 1
                next_items.append(next_item_one_hot)
        
        X_lstm = np.array(sequences)
        y_lstm = np.array(next_items)
        
        return (X_dnn, y_dnn), (X_lstm, y_lstm)
        
    def train(self, epochs=50, batch_size=32):
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 准备训练数据
        (X_dnn, y_dnn), (X_lstm, y_lstm) = self.prepare_training_data()
        
        # 训练DNN模型
        self.logger.info("\n训练DNN模型...")
        dnn_history = self.dnn_model.fit(
            X_dnn, y_dnn,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # 训练LSTM模型
        self.logger.info("\n训练LSTM模型...")
        lstm_history = self.lstm_model.fit(
            X_lstm, y_lstm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        history = {
            'dnn_history': dnn_history.history,  # 使用.history属性获取训练历史
            'lstm_history': lstm_history.history
        }
        return history  # 返回训练历史以便分析
        
    def recommend(self, user_id, n_recommendations=5):
        """为用户生成推荐"""
        # 获取用户历史
        user_history = self.ratings_df[self.ratings_df['user_id'] == user_id].sort_values('timestamp')
        
        if len(user_history) < self.seq_length:
            self.logger.info(f"用户 {user_id} 的历史记录不足，使用DNN模型推荐")
            return self._recommend_with_dnn(user_id, n_recommendations)
        
        # 使用LSTM模型预测
        try:
            recent_history = user_history.iloc[-self.seq_length:]
            sequence = np.zeros((1, self.seq_length, self.embedding_dim * 2))
            predictions = self.lstm_model.predict(sequence, verbose=0)[0]
            
            # 获取已观看电影
            watched_movies = user_history['item_id'].values
            
            # 过滤已观看电影
            predictions[watched_movies] = 0
            
            # 获取top-N推荐
            recommended_items = np.argsort(predictions)[-n_recommendations:][::-1]
            
            self.logger.info(f"成功为用户 {user_id} 生成推荐")
            return recommended_items
            
        except Exception as e:
            self.logger.error(f"LSTM推荐失败: {str(e)}")
            return self._recommend_with_dnn(user_id, n_recommendations)
            
    def _recommend_with_dnn(self, user_id, n_recommendations):
        """使用DNN模型生成推荐"""
        # 获取用户已观看电影
        watched_movies = self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ]['item_id'].values
        
        # 为所有未观看电影预测评分
        all_movies = np.arange(1, self.n_movies + 1)
        unwatched_movies = np.setdiff1d(all_movies, watched_movies)
        
        # 创建预测输入
        users = np.full_like(unwatched_movies, user_id)
        
        # 预测评分
        predictions = self.dnn_model.predict(
            [users, unwatched_movies],
            verbose=0
        )
        
        # 获取top-N推荐
        top_indices = np.argsort(predictions.flatten())[-n_recommendations:][::-1]
        recommended_items = unwatched_movies[top_indices]
        
        return recommended_items
        
    def evaluate(self, test_data):
        """评估模型性能"""
        self.logger.info("评估模型性能...")
        
        # 准备测试数据
        user_ids = test_data['user_id'].values
        movie_ids = test_data['item_id'].values
        actual_ratings = test_data['rating'].values
        
        # DNN预测
        predictions = self.dnn_model.predict(
            [user_ids, movie_ids],
            verbose=0
        )
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # 计算评估指标
        mse = np.mean((actual_ratings - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_ratings - predictions))
        
        self.logger.info(f"评估结果: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }

    def analyze_data(self):
        """数据分析与可视化"""
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 12})
        
        plt.figure(figsize=(18, 6))
        
        # 评分分布
        plt.subplot(1, 3, 1)
        sns.histplot(data=self.ratings_df, x='rating', bins=5)
        plt.title('Rating Distribution', fontsize=14, pad=10)
        plt.xlabel('Rating Value')
        plt.ylabel('Count')
        
        # 用户评分数量分布
        plt.subplot(1, 3, 2)
        user_rating_counts = self.ratings_df['user_id'].value_counts()
        sns.histplot(data=user_rating_counts, bins=30)
        plt.title('User Rating Distribution', fontsize=14, pad=10)
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        
        # 电影评分数量分布
        plt.subplot(1, 3, 3)
        movie_rating_counts = self.ratings_df['item_id'].value_counts()
        sns.histplot(data=movie_rating_counts, bins=30)
        plt.title('Movie Rating Distribution', fontsize=14, pad=10)
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Movies')
        
        plt.tight_layout()
        plt.savefig('hybrid_ratings_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_ablation_studies(self, test_data):
        """运行消融实验"""
        self.logger.info("=== 开始消融实验 ===")
        
        # 记录实验参数
        self.logger.info(f"初始参数: embedding_dim={self.embedding_dim}, lstm_units={self.lstm_units}, seq_length={self.seq_length}")
        
        # 测试不同的训练轮数
        self.logger.info("\n0. 测试不同的训练轮数:")
        epochs_list = [5, 10, 15, 20]
        epochs_results = []
        for epochs in epochs_list:
            self.logger.info(f"\n测试训练轮数 {epochs}")
            recommender = HybridRecommender(self.data_path)
            recommender.build_models()
            recommender.train(epochs=epochs, batch_size=32)
            metrics = recommender.evaluate(test_data)
            epochs_results.append((epochs, metrics))
            self.logger.info(f"训练轮数 = {epochs}: RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")
        
        # 1. 测试不同的嵌入维度
        self.logger.info("\n1. 测试不同的嵌入维度:")
        embedding_dims = [20, 50, 100]
        embedding_results = []
        for dim in embedding_dims:
            self.logger.info(f"\n测试嵌入维度 {dim}")
            recommender = HybridRecommender(self.data_path, embedding_dim=dim)
            recommender.build_models()
            recommender.train(epochs=15, batch_size=32)  # 使用15轮进行消融实验
            metrics = recommender.evaluate(test_data)
            embedding_results.append((dim, metrics))
            self.logger.info(f"嵌入维度 = {dim}: RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")
        
        # 2. 测试不同的LSTM单元数量
        self.logger.info("\n2. 测试不同的LSTM单元数量:")
        lstm_units = [16, 32, 64]
        lstm_results = []
        for units in lstm_units:
            self.logger.info(f"\n测试LSTM单元数量 {units}")
            recommender = HybridRecommender(self.data_path, lstm_units=units)
            recommender.build_models()
            recommender.train(epochs=15, batch_size=32)  # 使用15轮进行消融实验
            metrics = recommender.evaluate(test_data)
            lstm_results.append((units, metrics))
            self.logger.info(f"LSTM单元数量 = {units}: RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")
        
        # 3. 测试不同的序列长度
        self.logger.info("\n3. 测试不同的序列长度:")
        seq_lengths = [3, 5, 7]
        seq_results = []
        for length in seq_lengths:
            self.logger.info(f"\n测试序列长度 {length}")
            recommender = HybridRecommender(self.data_path)
            recommender.seq_length = length
            recommender.build_models()
            recommender.train(epochs=15, batch_size=32)  # 使用15轮进行消融实验
            metrics = recommender.evaluate(test_data)
            seq_results.append((length, metrics))
            self.logger.info(f"序列长度 = {length:3d}: RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}")
        
        return {
            'epochs_results': epochs_results,
            'embedding_results': embedding_results,
            'lstm_results': lstm_results,
            'seq_results': seq_results
        }

    def visualize_ablation_results(self, ablation_results):
        """可视化消融实验结果"""
        plt.figure(figsize=(20, 5))
        
        # 绘制训练轮数实验结果
        plt.subplot(1, 4, 1)
        epochs = [x[0] for x in ablation_results['epochs_results']]
        rmse = [x[1]['RMSE'] for x in ablation_results['epochs_results']]
        plt.plot(epochs, rmse, marker='o')
        plt.title('Impact of Training Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        
        # 绘制其他实验结果...
        
        plt.tight_layout()
        plt.savefig('ablation_results.png')

    def statistical_significance_test(self, results1, results2):
        """进行统计显著性测试"""
        # 执行t检验
        t_stat, p_value = stats.ttest_ind(results1, results2)
        return t_stat, p_value

def convert_to_serializable(obj):
    """将numpy类型转换为可JSON序列化的Python类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

def save_results(recommender, results_dict):
    """只保存结果，不重新运行实验"""
    # 将结果转换为可序列化的格式
    serializable_results = convert_to_serializable(results_dict)
    
    # 将结果保存为JSON文件
    with open('experiment_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    recommender.logger.info("实验结果已保存到 experiment_results.json")

def main():
    """运行实验并保存关键结果"""
    # 设置数据路径
    data_path = './ml-100k'
    
    # 创建推荐系统
    recommender = HybridRecommender(data_path)
    
    # 1. 生成数据分布可视化
    recommender.analyze_data()
    
    # 2. 构建和训练模型
    recommender.build_models()
    history = recommender.train()
    
    # 3. 准备测试数据
    _, test_data = train_test_split(
        recommender.ratings_df,
        test_size=0.2,
        random_state=42
    )
    
    # 4. 评估基准模型性能
    base_metrics = recommender.evaluate(test_data)
    
    # 5. 运行消融实验
    ablation_results = recommender.run_ablation_studies(test_data)
    
    # 6. 为示例用户生成推荐
    user_id = 1
    recommendations = recommender.recommend(user_id)
    
    # 7. 准备完整的结果字典
    results = {
        'dataset_stats': {
            'n_users': recommender.n_users,
            'n_movies': recommender.n_movies,
            'n_ratings': len(recommender.ratings_df),
            'rating_range': [recommender.ratings_df['rating'].min(), recommender.ratings_df['rating'].max()],
            'avg_rating': recommender.ratings_df['rating'].mean(),
            'sparsity': 1 - len(recommender.ratings_df) / (recommender.n_users * recommender.n_movies)
        },
        'training_history': history,
        'base_model_performance': base_metrics,
        'ablation_studies': ablation_results,
        'example_recommendations': {
            'user_id': user_id,
            'recommended_items': recommendations.tolist() if recommendations is not None else None
        }
    }
    
    # 8. 保存结果
    save_results(recommender, results)

if __name__ == "__main__":
    main() 