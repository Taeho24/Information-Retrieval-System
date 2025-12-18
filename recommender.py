import sys
import pandas as pd
import numpy as np
import os

class MatrixFactorization:
    def __init__(self, train_path, K=20, alpha=0.01, beta=0.02, epochs=20):
        """
        K: 잠재 요인(Latent Factor)의 개수
        alpha: 학습률 (Learning Rate)
        beta: 정규화 계수 (Regularization, 과적합 방지)
        epochs: 학습 반복 횟수
        """
        # 데이터 로드
        self.train_data = pd.read_csv(train_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 사용자 및 아이템 ID 매핑 (0부터 시작하는 인덱스로 변환)
        self.user_ids = self.train_data['user_id'].unique()
        self.item_ids = self.train_data['item_id'].unique()
        
        self.user_id_map = {id: i for i, id in enumerate(self.user_ids)}
        self.item_id_map = {id: i for i, id in enumerate(self.item_ids)}
        
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

        # 편향(Bias) 및 잠재 행렬 초기화
        # P: 사용자 잠재 행렬, Q: 아이템 잠재 행렬
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        self.b_u = np.zeros(self.num_users) # 사용자 편향
        self.b_i = np.zeros(self.num_items) # 아이템 편향
        self.b = np.mean(self.train_data['rating']) # 전체 평균
        
        # # 기본 통계 계산 (Cold Start 대비)
        # self.global_mean = self.train_data['rating'].mean()
        # self.user_means = self.train_data.groupby('user_id')['rating'].mean()
        # self.item_means = self.train_data.groupby('item_id')['rating'].mean()
        
        # # 사용자-아이템 매트릭스 생성 (피벗 테이블)
        # self.user_item_matrix = self.train_data.pivot(index='user_id', columns='item_id', values='rating')
        
        # # 평균 중심화(Mean Centering) - 사용자별 편향 제거
        # self.matrix_norm = self.user_item_matrix.subtract(self.user_item_matrix.mean(axis=1), axis=0)
        
        # # 사용자 간 유사도 계산 (코사인 유사도)
        # # NaN을 0으로 채우고 계산
        # self.user_similarity = self.compute_similarity(self.matrix_norm.fillna(0))

    # def compute_similarity(self, matrix):
    #     # Cosine Similarity: A . B / (|A| * |B|)
    #     # numpy를 사용하여 효율적으로 계산
    #     matrix_sparse = matrix.values
    #     similarity = np.dot(matrix_sparse, matrix_sparse.T)
        
    #     # 대각선 성분(자기 자신과의 내적)에 대한 제곱근 벡터
    #     square_mag = np.diag(similarity)
    #     inv_square_mag = 1 / square_mag
    #     inv_square_mag[np.isinf(inv_square_mag)] = 0
    #     inv_mag = np.sqrt(inv_square_mag)
        
    #     # 코사인 유사도 행렬 완성
    #     cosine_sim = similarity * inv_mag
    #     cosine_sim = cosine_sim.T * inv_mag
        
    #     return pd.DataFrame(cosine_sim, index=matrix.index, columns=matrix.index)

    def fit(self):
            # 학습 데이터를 리스트 형태로 변환 (속도 최적화)
            # (user_idx, item_idx, rating)
            self.samples = [
                (self.user_id_map[row['user_id']], self.item_id_map[row['item_id']], row['rating'])
                for _, row in self.train_data.iterrows()
            ]
            
            # SGD(Stochastic Gradient Descent) 학습 시작
            for epoch in range(self.epochs):
                np.random.shuffle(self.samples)
                
                for u, i, r in self.samples:
                    # 1. 현재 예측값 계산
                    prediction = self.b + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)
                    
                    # 2. 오차 계산 (실제값 - 예측값)
                    e = r - prediction
                    
                    # 3. 파라미터 업데이트 (경사 하강법)
                    # 편향 업데이트
                    self.b_u[u] += self.alpha * (e - self.beta * self.b_u[u])
                    self.b_i[i] += self.alpha * (e - self.beta * self.b_i[i])
                    
                    # 잠재 행렬 업데이트
                    # P[u] 업데이트를 위해 Q[i]가 필요하고, 그 반대도 마찬가지
                    P_u_old = self.P[u, :].copy() 
                    self.P[u, :] += self.alpha * (e * self.Q[i, :] - self.beta * self.P[u, :])
                    self.Q[i, :] += self.alpha * (e * P_u_old - self.beta * self.Q[i, :])
                
                # (옵션) 학습 진행 상황 출력 - 제출시에는 주석 처리 가능
                if (epoch+1) % 5 == 0:
                    print(f"Training: {2*(epoch+1)}% finished")
            
    def predict(self, user_id, item_id):
        """
        특정 사용자의 특정 아이템에 대한 평점을 예측
        """
        # 1. 학습 데이터에 있던 사용자와 아이템인 경우 -> MF 예측 수행
        if user_id in self.user_id_map and item_id in self.item_id_map:
            u = self.user_id_map[user_id]
            i = self.item_id_map[item_id]
            prediction = self.b + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)
            return prediction
            
        # 2. Cold Start (새로운 사용자/아이템) 처리
        elif user_id in self.user_id_map: # 기존 유저, 새로운 아이템
            u = self.user_id_map[user_id]
            return self.b + self.b_u[u]
        elif item_id in self.item_id_map: # 새로운 유저, 기존 아이템
            i = self.item_id_map[item_id]
            return self.b + self.b_i[i]
        else:
            return self.b

def main():
    # 인자 개수 확인
    if len(sys.argv) != 3:
        print("Usage: recommender.exe [training_file] [test_file]")
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # 출력 파일명 생성 (예: u1.base -> u1.base_prediction.txt)
    output_file = f"{train_file}_prediction.txt"

    print(f"Training Model on {train_file} ...")
    # model = Recommender(train_file)
    
    # 모델 생성 및 학습
    model = MatrixFactorization(train_file, K=40, alpha=0.004, beta=0.02, epochs=50)
    model.fit()
    
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
        
        # 평점 범위 1~5로 제한
        pred = np.clip(pred, 1, 5)
        
        results.append(f"{u_id}\t{i_id}\t{pred:.4f}")

    # 결과 파일 저장
    with open(output_file, 'w') as f:
        f.write('\n'.join(results) + '\n')
    
    print(f"Prediction saved to {output_file}")

if __name__ == "__main__":
    main()