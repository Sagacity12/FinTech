from operator import index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import  parser
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Exploratory Data Analysis
dataset = pd.read_csv('Dataset/FineTech_appData.csv')
# print(dataset.head())
# print(dataset.info())
# print(dataset.describe())

# Data Cleaning
dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)
print(dataset['hour'])

# Plotting
plotting = dataset.copy().drop(columns= ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
print(plotting)

# Histograms
plt.suptitle('Numerical Columns', fontsize=20)
for i in range(1, plotting.shape[1]+1):
    plt.subplot(3,3,i)
    f = plt.gca()
    f.set_title(plotting.columns.values[i - 1])

    vals = np.size(plotting.iloc[:, i - 1].unique())
    plt.hist(plotting.iloc[:, i - 1], bins = vals, color = 'blue')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# correlation with Response
corr = plotting.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                             title = 'Correlation With Response Variable',
                                             fontsize = 20, rot = 45, grid = True)
plt.show()
print(corr)

# correlation metrix
sn = sns.set(style = 'white', font_scale = 2)

# Compute the correlation matrix
corr = plotting.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (18,15))
f.suptitle('Correlation Matrix', fontsize=40)

# Generate a custom  diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask = mask, cmap = cmap, vmin = 0, vmax = 3, center = 0, square = True, linewidths= 5, cbar_kws = {"shrink": .5})
plt.show()

# feature engineering
# dataset['first_open'] = [parser.parse(row_data) for row_data in dataset['first_open']]
# dataset['enrolled_date'] = [parser.parse(row_data) for row_data in dataset['enrolled_date']]
dataset['first_open'] = pd.to_datetime(dataset['first_open'], errors='coerce')
dataset['enrolled_date'] = pd.to_datetime(dataset['enrolled_date'], errors='coerce')
print(dataset.dtypes)

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).dt.total_seconds() / 3600

plt.hist(dataset["difference"].dropna(), color = 'red', range = [0, 100])
plt.title('Difference Histogram', fontsize=20)
plt.show()

dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])
print(dataset)

# formatting the screen_list field
top_screens = pd.read_csv('Dataset/top_screens.csv')
print(top_screens.values)

dataset["screen_list"] = dataset.screen_list.astype(str) + ','
for sc in top_screens:
    dataset = dataset.screen_list.str.contains(sc).astype(int)
    dataset["screen_list"] = dataset.screen_list.str.replace(sc+",", "")
print(dataset["screen_list"])

# funnels
# savings_screens = [
#     "saving_1",
#     "saving_2",
#     "saving3Amount",
#     "saving_4",
#     "saving_5",
#     "saving_6",
#     "saving_7",
#     "saving_8",
#     "saving_9",
#     "saving_10",
# ]
#
# dataset["SavingsCount"] = dataset[savings_screens].sum(axis=1)
# dataset = dataset.drop(columns=savings_screens)
#
# print(dataset["SavingsCount"])

# Drop non-numeric columns before scaling
dataset = dataset.drop(columns=['screen_list'])

# Data Preprocessing
response = dataset["enrolled"]
dataset = dataset.drop(columns='enrolled')

X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size=0.2, random_state=42)

train_identifier = X_train['user']
X_train = X_train.drop(columns='user')
test_identifier = X_test['user']
X_test = X_test.drop(columns='user')

sc_X = StandardScaler()
X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(sc_X.transform(X_test), columns=X_test.columns, index=X_test.index)

# Model Building
classifier = LogisticRegression(random_state=42, penalty='l1', solver='liblinear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
plt.figure(figsize=(10, 6))
sns.set(font_scale=2)
sns.heatmap(df_cm, annot=True, annot_kws={"size": 20}, cmap="YlOrRd")
plt.show()

print("Test Data Accuracy: %0.2f" % accuracy_score(y_test, y_pred))

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Logistic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))