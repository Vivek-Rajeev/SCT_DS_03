import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


data = pd.read_csv('bank_data.csv', sep=';')  
data.columns = data.columns.str.strip('"')  

def preprocess_data(df, target_col='y'):
    
    df = df.fillna('unknown')
    
  
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_col:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    
    df[target_col] = df[target_col].map({'yes': 1, 'no': 0})
    
    
    df['pdays'] = df['pdays'].replace(999, 0)
    
    return df, label_encoders


data, le_dict = preprocess_data(data)


X = data.drop('y', axis=1)  
y = data['y']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20, 10))
plt.title("Decision Tree for Bank Marketing Prediction")
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, max_depth=3, impurity=False)
plt.show()


with open('model_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred))