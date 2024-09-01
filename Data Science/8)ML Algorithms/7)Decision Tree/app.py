import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

# Generate synthetic data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

# Sidebar inputs for DecisionTreeClassifier parameters
criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))
max_depth = st.sidebar.number_input('Max Depth', min_value=1, value=5)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, X_train.shape[0], 2, key=1234)
min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=1235)
max_features = st.sidebar.slider('Max Features', 1, 2, 2, key=1236)
max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes', min_value=0, value=0)
min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', min_value=0.0, value=0.0, step=0.01)

# Initial scatter plot
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    orig.empty()

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    # Initialize and train Decision Tree Classifier
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        random_state=42,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)

    # Plot decision boundary
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)

    st.subheader(f"Accuracy for Decision Tree: {accuracy_score(y_test, y_pred):.2f}")

    # Export decision tree as dot format and display it using graphviz_chart
    tree_graph = export_graphviz(clf, feature_names=["Col1", "Col2"], filled=True, rounded=True)
    st.graphviz_chart(tree_graph)
