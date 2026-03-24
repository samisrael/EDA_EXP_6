# EDA_EXP_6


**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

## Algorithm:

#### Step 1 - Load the wine dataset and check its basic structure, shape, and missing values.

#### Step 2 - Plot univariate distributions for alcohol, volatile acidity, and pH to understand individual feature behavior.

#### Step 3 - Create bivariate boxplots to study relationships between wine quality and key predictors.

#### Step 4 - Compute and visualize the correlation matrix to identify feature relationships with wine quality.

#### Step 5 - Convert wine quality into a binary good/bad label for classification.

#### Step 6 - Split the dataset into training and testing sets for model evaluation.

#### Step 7 - Train a Logistic Regression model and predict wine quality on test data.

#### Step 8 - Evaluate the model using accuracy and confusion matrix, and detect outliers using boxplots.

## Program:
```
Name : Sam Israel D
Reg No : 212222230128
```
```py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

1 - DATA UNDERSTANDING
print("First 5 rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

2 - UNIVARIATE ANALYSIS (HISTPLOTTING)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
sns.histplot(df['alcohol'], kde=True)
plt.title("Alcohol Distribution")

plt.subplot(1,3,2)
sns.histplot(df['volatile acidity'], kde=True)
plt.title("Volatile Acidity Distribution")

plt.subplot(1,3,3)
sns.histplot(df['pH'], kde=True)
plt.title("pH Distribution")

plt.tight_layout()
plt.show()


3 - BIVARIATE ANALYSIS
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol vs Quality")

plt.subplot(1,2,2)
sns.boxplot(x='quality', y='volatile acidity', data=df)
plt.title("Acidity vs Quality")

plt.tight_layout()
plt.show()

print("\nRelationship Explanation:")
print("- Higher quality wines tend to have higher alcohol levels.")
print("- Volatile acidity decreases as wine quality increases.")



4 - MULTIVARIATE ANALYSIS – CORRELATION

corr = df[['alcohol', 'volatile acidity', 'pH', 'quality']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nHighest Correlation with Quality:")
print(corr['quality'].sort_values(ascending=False))



5 - CLASSIFICATION – GOOD VS BAD WINE

df['good_wine'] = (df['quality'] >= 7).astype(int)

X = df.drop(['quality', 'good_wine'], axis=1)
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


6 - OUTLIER DETECTION

features = ['alcohol', 'pH', 'volatile acidity']

plt.figure(figsize=(12, 4))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[feature])
    plt.title(f"{feature} Boxplot")

plt.tight_layout()
plt.show()

```
## Output:
#### 1 - DATA UNDERSTANDING
<img width="997" height="355" alt="image" src="https://github.com/user-attachments/assets/8e022783-fc8f-4a68-8053-d2f7b7b0d5f9" />

<img width="497" height="117" alt="image" src="https://github.com/user-attachments/assets/925e86b2-e8c4-4245-be3d-5b93614713fc" />


#### 2 - UNIVARIATE ANALYSIS (HISTPLOTTING)

<img width="1463" height="382" alt="image" src="https://github.com/user-attachments/assets/d7f5ba47-b3b7-450b-97f0-8fb42ac13f91" />

#### 3 - BIVARIATE ANALYSIS

<img width="1239" height="380" alt="image" src="https://github.com/user-attachments/assets/0b407f06-3fb7-41a1-9ad1-a8d127d31896" />

#### 4 - MULTIVARIATE ANALYSIS – CORRELATION

<img width="1092" height="479" alt="image" src="https://github.com/user-attachments/assets/99435492-e0f9-459a-9dcb-73d2ab0b760a" />

#### 5 - CLASSIFICATION – GOOD VS BAD WINE

<img width="599" height="116" alt="image" src="https://github.com/user-attachments/assets/43cf4cba-cf6e-48e4-a0ab-35151e7699d9" />

#### 6 - OUTLIER DETECTION

<img width="1162" height="375" alt="image" src="https://github.com/user-attachments/assets/d4e9936d-dd13-4f24-91c4-9aa540986dad" />


## Result:
Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
