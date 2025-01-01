import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances

# 设置matplotlib参数
plt.rcParams['font.family'] = ['DejaVu Sans']  # 使用更通用的字体
plt.rcParams['axes.unicode_minus'] = False

class MovieRecommender:
    def __init__(self, data_path, k=10, use_normalization=True, similarity_metric='cosine'):
        self.data_path = data_path
        self.k = k  # 近邻用户数量
        self.use_normalization = use_normalization  # 是否使用评分标准化
        self.similarity_metric = similarity_metric  # 相似度计算方法
        self.ratings = None
        self.movies = None
        self.users = None
        self.user_movie_matrix = None
        self.user_similarity = None
        self.mean_ratings = None
        
    def load_data(self):
        """加载所有必要的数据文件"""
        # 加载评分数据
        self.ratings = pd.read_csv(f"{self.data_path}/u.data", 
                                 sep="\t", 
                                 names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 加载电影数据
        self.movies = pd.read_csv(f"{self.data_path}/u.item", 
                                sep="|", 
                                encoding='latin-1',
                                names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                      'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                      'Thriller', 'War', 'Western'])
        
        # 加载用户数据
        self.users = pd.read_csv(f"{self.data_path}/u.user", 
                               sep="|", 
                               names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        
    def preprocess_data(self):
        """数据预处理"""
        # 计算每个用户的平均评分
        self.mean_ratings = self.ratings.groupby('user_id')['rating'].mean()
        
        # 创建用户-电影评分矩阵
        self.user_movie_matrix = self.ratings.pivot(index='user_id', 
                                                  columns='item_id', 
                                                  values='rating')
        
        if self.use_normalization:
            # 对每个用户的评分减去其平均评分
            for user_id in self.user_movie_matrix.index:
                user_mean = self.mean_ratings[user_id]
                self.user_movie_matrix.loc[user_id] = self.user_movie_matrix.loc[user_id].fillna(user_mean)
                self.user_movie_matrix.loc[user_id] = self.user_movie_matrix.loc[user_id] - user_mean
        else:
            # 只填充缺失值，不进行标准化
            for user_id in self.user_movie_matrix.index:
                user_mean = self.mean_ratings[user_id]
                self.user_movie_matrix.loc[user_id] = self.user_movie_matrix.loc[user_id].fillna(user_mean)
        
        # 计算用户相似度
        if self.similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(self.user_movie_matrix)
        elif self.similarity_metric == 'pearson':
            self.user_similarity = self.user_movie_matrix.T.corr().values
        elif self.similarity_metric == 'euclidean':
            # 使用欧氏距离的倒数作为相似度
            euclidean_dist = euclidean_distances(self.user_movie_matrix)
            self.user_similarity = 1 / (1 + euclidean_dist)
        
    def analyze_data(self):
        """数据分析与可视化"""
        plt.figure(figsize=(15, 5))
        
        # 评分分布
        plt.subplot(1, 3, 1)
        sns.histplot(data=self.ratings, x='rating', bins=5)
        plt.title('Rating Distribution')
        
        # 用户评分数量分布
        plt.subplot(1, 3, 2)
        user_rating_counts = self.ratings['user_id'].value_counts()
        sns.histplot(data=user_rating_counts, bins=30)
        plt.title('User Rating Counts')
        
        # 电影评分数量分布
        plt.subplot(1, 3, 3)
        movie_rating_counts = self.ratings['item_id'].value_counts()
        sns.histplot(data=movie_rating_counts, bins=30)
        plt.title('Movie Rating Counts')
        
        plt.tight_layout()
        plt.savefig('ratings_analysis.png')
        plt.close()
        
    def predict_rating(self, user_id, item_id, k=10):
        """预测用户对特定电影的评分"""
        if user_id not in self.user_movie_matrix.index:
            return None
            
        # 获取最相似的k个用户
        user_similarities = self.user_similarity[user_id-1]
        similar_users = np.argsort(user_similarities)[::-1][1:k+1]
        
        # 计算预测评分
        similar_users_ratings = self.user_movie_matrix.iloc[similar_users][item_id]
        similar_users_similarities = user_similarities[similar_users]
        
        if np.sum(similar_users_similarities) == 0:
            return self.mean_ratings[user_id]
            
        # 预测评分（加上用户平均评分）
        predicted_rating = (np.sum(similar_users_ratings * similar_users_similarities) / 
                          np.sum(similar_users_similarities)) + self.mean_ratings[user_id]
        
        # 确保评分在1-5的范围内
        return max(1, min(5, predicted_rating))
        
    def recommend_movies(self, user_id, n_recommendations=5):
        """为用户推荐电影"""
        if user_id not in self.user_movie_matrix.index:
            return None
            
        # 获取用户已评分的电影
        rated_movies = self.ratings[self.ratings['user_id'] == user_id]['item_id'].values
        
        # 获取所有未评分的电影
        all_movies = self.movies['movie_id'].values
        unrated_movies = np.setdiff1d(all_movies, rated_movies)
        
        # 预测所有未评分电影的评分
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            if predicted_rating is not None:
                predictions.append((movie_id, predicted_rating))
                
        # 排序并获取推荐
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
        recommended_movies = pd.DataFrame(recommendations, columns=['movie_id', 'predicted_rating'])
        
        # 添加电影信息
        result = recommended_movies.merge(self.movies[['movie_id', 'movie_title']], on='movie_id')
        return result
        
    def evaluate_model(self, test_data):
        """评估模型性能"""
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            predicted = self.predict_rating(row['user_id'], row['item_id'])
            if predicted is not None:
                predictions.append(predicted)
                actuals.append(row['rating'])
                
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        return rmse, mae

def run_ablation_studies():
    """运行消融实验"""
    print("\n=== 开始消融实验 ===")
    
    # 准备测试数据
    test_data = pd.read_csv("ml-100k/u1.test", 
                           sep="\t", 
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # 1. 测试不同的K值
    k_values = [5, 10, 20, 50]
    print("\n1. 测试不同的K值:")
    for k in k_values:
        recommender = MovieRecommender("ml-100k", k=k)
        recommender.load_data()
        recommender.preprocess_data()
        rmse, mae = recommender.evaluate_model(test_data)
        print(f"K = {k:2d}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    
    # 2. 测试评分标准化的影响
    print("\n2. 测试评分标准化的影响:")
    for use_norm in [True, False]:
        recommender = MovieRecommender("ml-100k", use_normalization=use_norm)
        recommender.load_data()
        recommender.preprocess_data()
        rmse, mae = recommender.evaluate_model(test_data)
        print(f"使用标准化 = {use_norm}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    
    # 3. 测试不同的相似度计算方法
    print("\n3. 测试不同的相似度计算方法:")
    for sim_metric in ['cosine', 'pearson', 'euclidean']:
        recommender = MovieRecommender("ml-100k", similarity_metric=sim_metric)
        recommender.load_data()
        recommender.preprocess_data()
        rmse, mae = recommender.evaluate_model(test_data)
        print(f"{sim_metric:10s}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

if __name__ == "__main__":
    # 初始化推荐系统
    recommender = MovieRecommender("ml-100k")
    
    # 加载和预处理数据
    recommender.load_data()
    recommender.preprocess_data()
    
    # 数据分析
    recommender.analyze_data()
    
    # 为用户1推荐5部电影
    recommendations = recommender.recommend_movies(1, 5)
    print("\nRecommended movies for user 1:")
    print(recommendations)
    
    # 模型评估
    test_data = pd.read_csv("ml-100k/u1.test", 
                           sep="\t", 
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
    rmse, mae = recommender.evaluate_model(test_data)
    print(f"\nModel evaluation results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 运行消融实验
    run_ablation_studies() 