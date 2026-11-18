
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# =========================================
# ПРАКТИКА 10. КЛАСТЕРИЗАЦИЯ K-MEANS
# =========================================

# 1. ПАПКИ И ЗАГРУЗКА ДАННЫХ

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Mall_Customers.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("=== ИСХОДНЫЕ ДАННЫЕ ===")
print("Размер таблицы:", df.shape)
print(df.head())
print("\nПропущенные значения:")
print(df.isnull().sum())

# 2. ВЫБОР ПРИЗНАКОВ

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

if X.isnull().any().any():
    raise ValueError("В выбранных признаках есть пропуски")

# Масштабирование: приводим признаки
# к сопоставимому масштабу

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. МЕТОД ЛОКТЯ И СИЛУЭТ

k_values = list(range(2, 11))
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouettes.append(
        silhouette_score(X_scaled, labels)
    )

scores = pd.DataFrame({
    "k": k_values,
    "inertia": inertias,
    "silhouette": silhouettes
})

print("\n=== СРАВНЕНИЕ K ===")
print(scores.round(4).to_string(index=False))

scores.to_csv(
    RESULTS_DIR / "k_scores.csv",
    index=False
)

# 4. ГРАФИК МЕТОДА ЛОКТЯ

plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    inertias,
    marker="o"
)
plt.xticks(k_values)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "01_elbow.png",
    dpi=150
)
plt.show()

# 5. ГРАФИК СИЛУЭТА

plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    silhouettes,
    marker="o",
    color="green"
)
plt.xticks(k_values)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "02_silhouette.png",
    dpi=150
)
plt.show()

best_silhouette_k = int(
    scores.loc[scores["silhouette"].idxmax(), "k"]
)

print("\nЛучшее k по силуэту:", best_silhouette_k)

# 6. ВЫБОР ИТОГОВОГО K
# По умолчанию используем максимум силуэта.
# После изучения графика локтя можно
# обоснованно изменить selected_k вручную.

selected_k = 5

print("Итоговое выбранное k:", selected_k)

# 7. ОБУЧЕНИЕ K-MEANS

kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n=== КОЛИЧЕСТВО КЛИЕНТОВ ===")
print(df["Cluster"].value_counts().sort_index())

# 8. ПРОФИЛИ КЛАСТЕРОВ

profiles = df.groupby("Cluster")[features].mean()
profiles["Count"] = df.groupby("Cluster").size()

# Сначала присваиваем понятные названия
# на основе относительных средних значений.

overall = df[features].mean()

def cluster_name(row):
    income = (
        "High income"
        if row["Annual Income (k$)"] >= overall["Annual Income (k$)"]
        else "Low income"
    )

    spending = (
        "High spending"
        if row["Spending Score (1-100)"] >= overall["Spending Score (1-100)"]
        else "Low spending"
    )

    return income + " / " + spending

profiles["Segment"] = profiles.apply(
    cluster_name,
    axis=1
)

# Если два кластера имеют одинаковое название,
# добавляем их средний возраст для различения.

duplicates = profiles["Segment"].duplicated(
    keep=False
)

for cluster_id in profiles.index[duplicates]:
    age = profiles.loc[cluster_id, "Age"]
    profiles.loc[cluster_id, "Segment"] += (
        f" / Avg age {age:.0f}"
    )

print("\n=== ПРОФИЛИ КЛАСТЕРОВ ===")
print(profiles.round(2).to_string())

profiles.round(2).to_csv(
    RESULTS_DIR / "cluster_profiles.csv"
)

name_map = profiles["Segment"].to_dict()
df["Segment"] = df["Cluster"].map(name_map)

df.to_csv(
    RESULTS_DIR / "customers_with_clusters.csv",
    index=False
)

# 9. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=65
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "03_clusters.png",
    dpi=150
)
plt.show()

# 10. ТЕПЛОВАЯ КАРТА ПРОФИЛЕЙ

profile_scaled = pd.DataFrame(
    scaler.transform(profiles[features]),
    index=profiles.index,
    columns=features
)

plt.figure(figsize=(9, 5))
sns.heatmap(
    profile_scaled,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Cluster Profiles (Standardized)")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "04_profiles_heatmap.png",
    dpi=150
)
plt.show()

# 11. ПРОВЕРКА УСТОЙЧИВОСТИ

kmeans2 = KMeans(
    n_clusters=selected_k,
    random_state=123,
    n_init=10
)

labels2 = kmeans2.fit_predict(X_scaled)

ari = adjusted_rand_score(
    df["Cluster"],
    labels2
)

print("\n=== ПРОВЕРКА УСТОЙЧИВОСТИ ===")
print("Первый random_state: 42")
print("Второй random_state: 123")
print("Adjusted Rand Index:", round(ari, 4))

if np.isclose(ari, 1.0):
    print("Состав кластеров не изменился.")
else:
    print("Состав кластеров изменился.")

# Номера кластеров могут поменяться местами.
# ARI учитывает это и сравнивает состав групп.

comparison = pd.crosstab(
    df["Cluster"],
    labels2,
    rownames=["Seed 42"],
    colnames=["Seed 123"]
)

print("\nТаблица соответствия кластеров:")
print(comparison)

comparison.to_csv(
    RESULTS_DIR / "stability_comparison.csv"
)

with open(
    RESULTS_DIR / "summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("Practice 10 - KMeans\n")
    f.write(f"Best silhouette k: {best_silhouette_k}\n")
    f.write(f"Selected k: {selected_k}\n")
    f.write(f"ARI: {ari:.4f}\n\n")
    f.write("Cluster profiles:\n")
    f.write(profiles.round(2).to_string())

print("\nГОТОВО!")
print("Все результаты сохранены в папке results.")
