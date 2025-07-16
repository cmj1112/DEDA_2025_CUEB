# 开发人员:xiaol
# 开发时间:2025/7/13 15:41
# 文件名称:HW2_4.PY
# 开发工具:PyCharm

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                           confusion_matrix,
                           ConfusionMatrixDisplay,
                           classification_report)

# 1. Generate synthetic data (3-class problem)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. Create Random Forest model
rf_clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# 4. Train model
rf_clf.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = rf_clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 6. Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1", "Class 2"])
disp.plot(cmap='Blues', ax=ax, values_format='d')  # Show integer values
plt.title("Random Forest Confusion Matrix", pad=20, fontsize=14)
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Feature importance visualization (horizontal bar plot)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(
    rf_clf.feature_importances_,
    index=[f"Feature_{i}" for i in range(X.shape[1])]
)
feat_importances.nlargest(10).plot(kind='barh', color='steelblue')
plt.title("Top 10 Important Features", pad=15, fontsize=14)
plt.xlabel("Feature Importance Score", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()