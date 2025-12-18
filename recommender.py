import sys
import pandas as pd
import numpy as np
import os

class Recommender:
    def __init__(self, train_path):
        
        # 데이터 로드
        self.train_data = pd.read_csv(train_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 기본 통계 계산 (Cold Start 대비)
        self.global_mean = self.train_data['rating'].mean()
        self.user_means = self.train_data.groupby('user_id')['rating'].mean()
        self.item_means = self.train_data.groupby('item_id')['rating'].mean()
        
        # 사용자-아이템 매트릭스 생성 (피벗 테이블)
        self.user_item_matrix = self.train_data.pivot(index='user_id', columns='item_id', values='rating')
        
        # 평균 중심화(Mean Centering) - 사용자별 편향 제거
        self.matrix_norm = self.user_item_matrix.subtract(self.user_item_matrix.mean(axis=1), axis=0)
        
        # 사용자 간 유사도 계산 (코사인 유사도)
        # NaN을 0으로 채우고 계산
        self.user_similarity = self.compute_similarity(self.matrix_norm.fillna(0))

    def compute_similarity(self, matrix):
        # Cosine Similarity: A . B / (|A| * |B|)
        # numpy를 사용하여 효율적으로 계산
        matrix_sparse = matrix.values
        similarity = np.dot(matrix_sparse, matrix_sparse.T)
        
        # 대각선 성분(자기 자신과의 내적)에 대한 제곱근 벡터
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        
        # 코사인 유사도 행렬 완성
        cosine_sim = similarity * inv_mag
        cosine_sim = cosine_sim.T * inv_mag
        
        return pd.DataFrame(cosine_sim, index=matrix.index, columns=matrix.index)

    def predict(self, user_id, item_id, k_neighbors=20):
        """
        특정 사용자의 특정 아이템에 대한 평점을 예측
        """
        # 1. 학습 데이터에 없는 아이템인 경우 -> 유저 평균 or 글로벌 평균
        if item_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.global_mean)

        # 2. 학습 데이터에 없는 유저인 경우 -> 아이템 평균 or 글로벌 평균
        if user_id not in self.user_item_matrix.index:
            return self.item_means.get(item_id, self.global_mean)
        
        # 3. 협업 필터링 예측
        # 해당 아이템을 평가한 다른 사용자들을 찾음
        users_rated_item = self.user_item_matrix[item_id].dropna().index
        
        # 유사도 가져오기
        sim_scores = self.user_similarity.loc[user_id, users_rated_item]
        
        # 유사도가 높은 상위 K명의 이웃 선정
        top_k_users = sim_scores.nlargest(k_neighbors)
        
        # 이웃이 없거나 유사도가 모두 0 이하인 경우 -> 아이템 평균으로 대체
        if top_k_users.empty or top_k_users.sum() <= 0:
            return self.item_means.get(item_id, self.global_mean)
        
        # 예측 평점 계산: (유사도 * 평점)의 합 / 유사도의 합
        ratings_k = self.user_item_matrix.loc[top_k_users.index, item_id]
        weighted_sum = np.dot(top_k_users, ratings_k)
        sum_of_weights = top_k_users.sum()
        
        predicted_rating = weighted_sum / sum_of_weights
        
        # 평점 범위(1~5)로 클리핑
        return np.clip(predicted_rating, 1, 5)

def main():
    # 인자 개수 확인
    if len(sys.argv) != 3:
        print("Usage: recommender.exe [training_file] [test_file]")
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # 출력 파일명 생성 (예: u1.base -> u1.base_prediction.txt)
    output_file = f"{train_file}_prediction.txt"

    print(f"Loading data from {train_file}...")
    model = Recommender(train_file)
    
    print(f"Predicting for {test_file}...")
    test_data = pd.read_csv(test_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # 예측 수행 및 결과 저장
    results = []
    
    # 테스트 데이터의 각 행에 대해 예측
    # iterrows는 느리지만 코드가 직관적입니다. 속도가 중요하다면 vectorization을 고려해야 합니다.
    for _, row in test_data.iterrows():
        u_id = int(row['user_id'])
        i_id = int(row['item_id'])
        
        pred = model.predict(u_id, i_id)
        results.append(f"{u_id}\t{i_id}\t{pred:.4f}")

    # 결과 파일 저장
    with open(output_file, 'w') as f:
        f.write('\n'.join(results) + '\n') # 마지막 줄바꿈 포함
    
    print(f"Prediction saved to {output_file}")

if __name__ == "__main__":
    main()