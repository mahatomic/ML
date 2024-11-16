import numpy as np
import matplotlib.pyplot as plt


def generate_data(Xmin1, Xmax1, Xmin2, Xmax2, num_points):
    """Генерация случайных двумерных векторов данных."""
    return np.random.uniform([Xmin1, Xmin2], [Xmax1, Xmax2], (num_points, 2))


def initialize_centroids(data, num_clusters):
    """Случайный выбор центров кластеров из данных."""
    indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    return data[indices]


def euclidean_distance(a, b):
    """Вычисление евклидова расстояния."""
    return np.sqrt(np.sum((a - b) ** 2))


def assign_clusters(data, centroids):
    """Классификация данных по ближайшему центру кластера."""
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)


def update_centroids(data, clusters, num_clusters):
    """Обновление центров кластеров."""
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(num_clusters)])
    return new_centroids


def kmeans_clustering(data, num_clusters, epsilon=1e-4, max_iterations=100):
    """Алгоритм K-средних для кластеризации."""
    centroids = initialize_centroids(data, num_clusters)
    prev_error = float('inf')

    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, num_clusters)

        # Вычисление средних расстояний
        distances = [euclidean_distance(data[i], centroids[clusters[i]]) for i in range(len(data))]
        current_error = np.mean(distances)

        # Проверка условия окончания
        if abs(prev_error - current_error) <= epsilon:
            break

        prev_error = current_error
        centroids = new_centroids

    return clusters, centroids


# 1. Неструктурированные данные

# Параметры генерации неструктурированных данных
Xmin1, Xmax1 = 0, 10  # Диапазон для x1
Xmin2, Xmax2 = 0, 10  # Диапазон для x2
num_points = 100  # Количество векторов
num_clusters = 5  # Количество кластеров

# Генерация данных
data = generate_data(Xmin1, Xmax1, Xmin2, Xmax2, num_points)

# Распределение векторов на плоскости
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.6)
plt.title('Распределение двумерных векторов данных')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(Xmin1, Xmax1)
plt.ylim(Xmin2, Xmax2)
plt.grid()
plt.show()

# Кластеризация
clusters, centroids = kmeans_clustering(data, num_clusters)

# Визуализация результатов
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    plt.scatter(data[clusters == i][:, 0], data[clusters == i][:, 1], label=f'Кластер {i + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Центры кластеров')
plt.title('Кластеризация двумерных векторов данных (Неструктурированные данные)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(Xmin1, Xmax1)
plt.ylim(Xmin2, Xmax2)
plt.grid()
plt.legend()
plt.show()


# 2. Структурированные данные

def nngenc(bounds, clusters, points, std_dev):
    """Генерация двумерных векторов данных с кластерной структурой в заданном диапазоне."""
    data = []
    for _ in range(clusters):
        # Генерация центра кластера
        center = np.random.uniform(bounds[0], bounds[1], 2)
        # Генерация точек вокруг центра с заданным стандартным отклонением
        cluster_points = np.random.normal(loc=center, scale=std_dev, size=(points, 2))
        data.append(cluster_points)
    return np.vstack(data)


# Параметры генерации структурированных данных
bounds = [0, 1]  # Диапазон
clusters = 5  # Количество кластеров
points = 100  # Число векторов в каждом кластере
std_dev = 0.05  # Стандартное отклонение векторов в кластере

# Генерация данных
structured_data = nngenc(bounds, clusters, points, std_dev)

# Распределение векторов на плоскости
plt.figure(figsize=(10, 6))
plt.scatter(structured_data[:, 0], structured_data[:, 1], color='red', alpha=0.6)
plt.title('Распределение двумерных векторов данных с кластерной структурой')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(bounds[0], bounds[1])
plt.ylim(bounds[0], bounds[1])
plt.grid()
plt.legend()
plt.show()

# Кластеризация
structured_clusters, structured_centroids = kmeans_clustering(structured_data, clusters)

# Визуализация результатов
plt.figure(figsize=(10, 6))
for i in range(clusters):
    plt.scatter(structured_data[structured_clusters == i][:, 0], structured_data[structured_clusters == i][:, 1],
                label=f'Кластер {i + 1}')
plt.scatter(structured_centroids[:, 0], structured_centroids[:, 1], color='red', marker='X', s=200,
            label='Центры кластеров')
plt.title('Кластеризация двумерных векторов данных (Структурированные данные)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(bounds[0], bounds[1])
plt.ylim(bounds[0], bounds[1])
plt.grid()
plt.legend()
plt.show()
