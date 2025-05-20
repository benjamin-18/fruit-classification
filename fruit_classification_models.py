import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, KFold
#from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
#from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

here = os.path.dirname(__file__)
source_0 = os.path.join(here, "apple")
source_1 = os.path.join(here, "orange")

image_size = (32, 32)
# We downsize to reduce the number of independent variables.

#### IMPORTING AND PREPROCESSING IMAGES.

def load_and_flatten_image(path):
    img = Image.open(path).convert('RGB').resize(image_size)
    img_array = np.asarray(img).astype(np.float32) / 255  # Normalize RGB pixel values, which span from 0 to 255, to be in [0,1].
    return img_array.flatten()

X = []
y = []

for file in os.listdir(source_0):
    if file.lower().endswith(('.jpg', '.jpeg')): # Ensure that it only reads jpegs.
        X.append(load_and_flatten_image(os.path.join(source_0, file))) # Add the reduced RGB values to the input data X.
        y.append(0) # Add 0 to classes y.

for file in os.listdir(source_1):
    if file.lower().endswith(('.jpg', '.jpeg')):
        X.append(load_and_flatten_image(os.path.join(source_1, file)))
        y.append(1)

X = pd.DataFrame(X)
y = pd.Series(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use PCA
pca = PCA(n_components=50, random_state=1)
X = pca.fit_transform(X)

#### MACHINE LEARNING MODELS.

### LOGISTIC REGRESSION.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

log_reg = LogisticRegression(max_iter = 5000).fit(X_train,y_train)

yhat = log_reg.predict(X_test)
yhat_prob = log_reg.predict_proba(X_test)

logreg_comparison_predictions = pd.DataFrame({'yhat_prob': yhat, 'y_test': y_test})
logreg_comparison_probs = pd.DataFrame({'yhat_prob': yhat_prob[:, 1], 'y_test': y_test})

logreg_correct = 0
for i in range(0, len(logreg_comparison_predictions)):
    if logreg_comparison_predictions.iloc[i, 0] == logreg_comparison_predictions.iloc[i, 1]:
        logreg_correct += 1
logreg_correct = logreg_correct / len(logreg_comparison_predictions)
print("LOGISTIC REGRESSION: The proportion of correct predictions is",logreg_correct,".")
# ADD A BETTER EVALUATION METRIC.

# Cross validation.
kf = KFold(n_splits=5, shuffle=True, random_state=1)

log_reg = LogisticRegression(max_iter = 5000)
logreg_scores = cross_val_score(log_reg, X, y, cv=kf)

print("Cross-validation scores:", logreg_scores)
print("Mean accuracy:", logreg_scores.mean())
print("Standard deviation:", logreg_scores.std())

### DECISION TREE.

tree_model = DecisionTreeClassifier(criterion="entropy", max_depth = 3, random_state=1)
tree_scores = cross_val_score(tree_model, X, y, cv=kf)

print("\nDECISION TREE")
print("Cross-validation scores:", tree_scores)
print("Mean accuracy:", tree_scores.mean())
print("Standard deviation:", tree_scores.std())

# For visualisation:
tree_model.fit(X, y)
plot_tree(tree_model)
plt.title("Decision Tree Classifier")
plt.savefig("decision_tree_plot.pdf", format='pdf')
plt.show()

### RANDOM FORESTS.

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1)

random_forest_scores = cross_val_score(random_forest_model, X, y, cv=kf)

print("\nRANDOM FOREST")
print("Cross-validation scores:", random_forest_scores)
print("Mean accuracy:", random_forest_scores.mean())
print("Standard deviation:", random_forest_scores.std())
# What hapepns when you increase n_estimators?

### SUPPORT VECTOR MACHINES.

SVM_model = LinearSVC(class_weight='balanced', random_state=1, fit_intercept=False, max_iter = 5000)

SVM_scores = cross_val_score(SVM_model, X, y, cv=kf)
print("\nSUPPORT VECTOR MACHINES")
print("Cross-validation scores:", SVM_scores)
print("Mean accuracy:", SVM_scores.mean())
print("Standard deviation:", SVM_scores.std())

### K NEAREST NEIGHBOURS.

KNN_model = KNeighborsClassifier(n_neighbors=5)

KNN_scores = cross_val_score(KNN_model, X, y, cv=kf)
print("\nK-NEAREST NEIGHBOURS")
print("Cross-validation scores:", KNN_scores)
print("Mean accuracy:", KNN_scores.mean())
print("Standard deviation:", KNN_scores.std())


