import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical

os.makedirs('images', exist_ok=True)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Cell_ID', 'Batch_ID', 'Defect_Type', 'Inspector_Comment'])
    df = df.dropna()

    X = df.drop(columns=['QC_Grade'])
    y = df['QC_Grade']

    categorical_cols = ['Production_Line', 'Shift', 'Supplier']
    numerical_cols = ['Ambient_Temp_C', 'Anode_Overhang_mm', 'Electrolyte_Volume_ml', 
                      'Internal_Resistance_mOhm', 'Capacity_mAh', 'Retention_50Cycle_Pct']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    return X, y, preprocessor, numerical_cols, categorical_cols

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig(f'images/{filename}', dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, filename):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:] # Pega as 10 mais importantes
    
    plt.figure(figsize=(8, 6))
    plt.title('Importância das Variáveis - Árvore de Decisão')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    plt.savefig(f'images/{filename}', dpi=300)
    plt.close()

def train_naive_bayes(X_train, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_neural_network(X_train, y_train_encoded, num_classes):
    y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
    ann_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ann_model.fit(X_train, y_train_cat, epochs=50, batch_size=32, verbose=0)
    return ann_model

def main():
    filepath = 'data/ev_battery_qc_data_2026_kaggle.csv'
    X, y, preprocessor, num_cols, cat_cols = load_and_preprocess_data(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    classes_labels = sorted(y.unique())

    # Naive Bayes
    nb_model = train_naive_bayes(X_train_processed, y_train)
    y_pred_nb = nb_model.predict(X_test_processed)
    print("--- Naive Bayes ---")
    print(classification_report(y_test, y_pred_nb))
    plot_confusion_matrix(y_test, y_pred_nb, classes_labels, 'Matriz de Confusão - Naive Bayes', 'cm_naive_bayes.png')

    # Decision Tree
    dt_model = train_decision_tree(X_train_processed, y_train)
    y_pred_dt = dt_model.predict(X_test_processed)
    print("\n--- Decision Tree ---")
    print(classification_report(y_test, y_pred_dt))
    plot_confusion_matrix(y_test, y_pred_dt, classes_labels, 'Matriz de Confusão - Decision Tree', 'cm_decision_tree.png')
   
    encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_cols)
    plot_feature_importance(dt_model, all_feature_names, 'feature_importance_dt.png')

    # SVM
    svm_model = train_svm(X_train_processed, y_train)
    y_pred_svm = svm_model.predict(X_test_processed)
    print("\n--- Support Vector Machine ---")
    print(classification_report(y_test, y_pred_svm))
    plot_confusion_matrix(y_test, y_pred_svm, classes_labels, 'Matriz de Confusão - SVM', 'cm_svm.png')

    # Neural Network
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    num_classes = len(encoder.classes_)
    
    ann_model = train_neural_network(X_train_processed, y_train_encoded, num_classes)
    y_pred_prob = ann_model.predict(X_test_processed, verbose=0)
    y_pred_ann = np.argmax(y_pred_prob, axis=1)
    
    print("\n--- Artificial Neural Network ---")
    print(classification_report(y_test_encoded, y_pred_ann, target_names=encoder.classes_))
    plot_confusion_matrix(y_test_encoded, y_pred_ann, encoder.classes_, 'Matriz de Confusão - Rede Neural', 'cm_neural_network.png')

    # Validação Cruzada
    print("\n--- Cross-Validation (K-Fold = 5) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores_nb = cross_val_score(GaussianNB(), X_train_processed, y_train, cv=cv, scoring='accuracy')
    print(f"Naive Bayes CV Accuracy: {cv_scores_nb.mean():.2f} (+/- {cv_scores_nb.std() * 2:.2f})")

    cv_scores_dt = cross_val_score(DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42), X_train_processed, y_train, cv=cv, scoring='accuracy')
    print(f"Decision Tree CV Accuracy: {cv_scores_dt.mean():.2f} (+/- {cv_scores_dt.std() * 2:.2f})")

    cv_scores_svm = cross_val_score(SVC(kernel='linear', random_state=42), X_train_processed, y_train, cv=cv, scoring='accuracy')
    print(f"SVM CV Accuracy: {cv_scores_svm.mean():.2f} (+/- {cv_scores_svm.std() * 2:.2f})")

if __name__ == "__main__":
    main()