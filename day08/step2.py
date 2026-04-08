import numpy as np
import cv2
import mnist  # 사용자 정의 모듈 (mnist.py가 같은 경로에 있어야 함)
from matplotlib import pyplot as plt

# ① 훈련 데이터와 테스트 데이터 로드
train, train_labels = mnist.getTrain()
test, test_labels = mnist.getTest()

print(f"훈련 데이터: {train.shape[0]}개 샘플")
print(f"테스트 데이터: {test.shape[0]}개 샘플")

# ② k-NN 모델 생성 및 훈련
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
print("✅ k-NN 모델 훈련 완료")

# ③ [과제 2-1] k값을 1~20까지 변경하며 정확도 측정
print("\n" + "="*50)
print("k값에 따른 정확도 비교 (1~20)")
print("="*50)

accuracies = {}

for k in range(1, 21):
    ret, result, neighbors, distance = knn.findNearest(test, k=k)
    
    correct = np.sum(result == test_labels)
    accuracy = correct / result.size * 100.0
    
    accuracies[k] = accuracy
    print(f"k={k:2d}: {accuracy:.2f}% ({correct}/{result.size})")

# 최적 k값 결과 출력
best_k = max(accuracies, key=accuracies.get)
best_acc = accuracies[best_k]

print("-" * 50)
print(f" 최적 k값: {best_k}")
print(f" 최고 정확도: {best_acc:.2f}%")
print("=" * 50)

# ④ [과제 2-2] 테스트 샘플 시각화
# 2행 5열의 서브플롯 생성
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('MNIST Test Samples Visualization', fontsize=16)

for i in range(10):
    ax = axes[i // 5, i % 5]
    
    # 1차원 배열(400개 특징)을 2D 이미지(20x20)로 변환
    img = test[i].reshape(20, 20)
    
    # 이미지 표시 (흑백 모드)
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {int(test_labels[i].item())}")
    ax.axis('off')  # 축 눈금 숨기기

plt.tight_layout()
plt.show()