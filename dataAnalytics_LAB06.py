from sklearn.neural_network import MLPClassifier #用於建立多層感知器模型。
from sklearn.datasets import fetch_openml #用於取得MNIST資料集
from sklearn.model_selection import train_test_split #分成訓練集和測試集
from sklearn.metrics import accuracy_score #計算分類模型的準確率

mnist = fetch_openml('mnist_784', as_frame=False, cache=False)

X = mnist.data # 特徵
Y = mnist.target.astype('int') # 目標

# 調整為 28x28
X = X.reshape(-1, 28, 28)

# 像素值縮放到 0 到 1 之間，以便模型更好地處理資料
X = X / 255.0

# 將圖像資料展平為一維數組，以便能夠輸入到多層感知器模型
X = X.reshape(X.shape[0], -1)

# 將資料集切分為訓練集和測試集，其中測試集佔總資料集的 10000 個樣本，random_state=42 用於設置隨機種子
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10000, random_state=42)

# 創建MLP分類器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=42,
                    learning_rate_init=0.1)

# 將訓練集的特徵 X_train 和對應的目標標籤 y_train 輸入到模型中
mlp.fit(X_train, Y_train)

# 預測，並計算模型在測試集上的準確率
predictions = mlp.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

train_accuracy = mlp.score(X_train, Y_train)
test_accuracy = mlp.score(X_test, Y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

plt.figure(figsize=(8, 6))
plt.plot(mlp.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
