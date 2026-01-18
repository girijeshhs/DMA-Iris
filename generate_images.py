import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (assuming it's in the same directory or adjust the path)
# For local testing, you might need to adjust this path
try:
    df = pd.read_csv("Iris.csv")
except:
    # If Iris.csv is not found, try loading from seaborn's built-in dataset
    df = sns.load_dataset('iris')
    # Rename columns to match your format
    df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    df['Id'] = range(1, len(df) + 1)

num = df.drop(["Id", "Species"], axis=1)

# 1. Bar Chart
plt.figure(figsize=(10, 6))
num.mean().plot(kind="bar", color='steelblue')
plt.title("Bar Chart - Mean of Iris Features", fontsize=16, fontweight='bold')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Mean Value", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/1_bar_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Line Chart
plt.figure(figsize=(10, 6))
plt.plot(num["SepalLengthCm"], linewidth=2, color='green')
plt.title("Line Chart - Sepal Length", fontsize=16, fontweight='bold')
plt.xlabel("Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/2_line_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Histogram
plt.figure(figsize=(10, 6))
plt.hist(num["PetalLengthCm"], bins=15, color='coral', edgecolor='black')
plt.title("Histogram - Petal Length", fontsize=16, fontweight='bold')
plt.xlabel("Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig("images/3_histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(num["SepalLengthCm"], num["PetalLengthCm"], alpha=0.6, c='purple', s=50)
plt.title("Scatter Plot - Sepal vs Petal Length", fontsize=16, fontweight='bold')
plt.xlabel("Sepal Length", fontsize=12)
plt.ylabel("Petal Length", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/4_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Box Plot
plt.figure(figsize=(10, 6))
plt.boxplot(num.values, labels=num.columns)
plt.title("Box Plot - Feature Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/5_box_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Heatmap
plt.figure(figsize=(10, 8))
corr = num.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', square=True, linewidths=1)
plt.title("Heatmap - Correlation Matrix", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("images/6_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Pair Plot
pairplot = sns.pairplot(df, hue="Species", palette='Set1', height=2.5)
pairplot.fig.suptitle("Pair Plot - Feature Relationships by Species", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("images/7_pair_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. K-Means Cluster Plot
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(num)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(num["PetalLengthCm"], num["PetalWidthCm"], c=clusters, cmap="viridis", s=100, alpha=0.6, edgecolors='black')
plt.colorbar(scatter, label='Cluster')
plt.title("Cluster Visualization (K-Means)", fontsize=16, fontweight='bold')
plt.xlabel("Petal Length", fontsize=12)
plt.ylabel("Petal Width", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/8_kmeans_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. Facet Plot
g = sns.FacetGrid(df, col="Species", height=4, aspect=1)
g.map(plt.scatter, "SepalLengthCm", "PetalLengthCm", alpha=0.6, s=50)
g.set_axis_labels("Sepal Length (cm)", "Petal Length (cm)")
g.fig.suptitle("Facet Plot - Species Comparison", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("images/9_facet_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ All images generated successfully in the 'images' folder!")
