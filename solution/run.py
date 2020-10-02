### IMPORT LIBRARIES
##### Built in libraries
import os

###### Use pip or anaconda to install
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

### READ DATA
##### Needs to be ran from the project directory
csv_measurements = os.path.join(os.getcwd(), 'datasets', 'measurements.csv')
df_measurements = pd.read_csv(csv_measurements, parse_dates=['measurement_time'])
df_measurements = df_measurements.sort_values(by=['measurement_time'], ascending=[True])

csv_failures = os.path.join(os.getcwd(), 'datasets', 'failures.csv')
df_failures = pd.read_csv(csv_failures, parse_dates=['failure_time'])
df_failures = df_failures.sort_values(by=['failure_time'], ascending=[True])

### MERGE NEXT FAILURE TO MEASUREMENTS
df_combined = pd.merge_asof(
    df_measurements,
    df_failures,
    left_on='measurement_time',
    right_on='failure_time',
    by='gadget_id',
    direction='forward',
)

### TRANSFORM COLUMNS
df_combined['time_to_fail'] = df_combined['failure_time']-df_combined['measurement_time']
df_combined['fail_in_1h'] = np.where(df_combined['time_to_fail']<pd.Timedelta(hours=1), 1, 0)

### CALCULATE RUNNING MEASURES
df_combined = df_combined.reset_index(drop=True)
df_combined = df_combined.sort_values(by=['gadget_id', 'measurement_time'], ascending=[True, True])

df_combined['temperature_6h_std'] = df_combined.groupby('gadget_id')['temperature'].rolling(6).std(ddof=0).reset_index(drop=True)
df_combined['pressure_6h_mean'] = df_combined.groupby('gadget_id')['pressure'].rolling(6).mean().reset_index(drop=True)

### SPLIT TO TRAIN AND TEST
X = ['vibration_y', 'pressure_6h_mean', 'temperature_6h_std']
y = 'fail_in_1h'
cols = X + [y]

df_to_split = df_combined.copy()
df_to_split = df_to_split.dropna(subset=cols)
df_to_split = df_to_split.reset_index(drop=True)

##### Create binary bins to 
binner = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='kmeans')
binner.fit(df_to_split[X])
arr_bins= binner.transform(df_to_split[X])
df_bins = pd.DataFrame(arr_bins)

X = list(df_bins.columns)
cols = X + [y]

df_to_split = pd.concat([df_to_split, df_bins], axis=1)

df_train = df_to_split[df_to_split['gadget_id'].isin([1,2,3,4])].reset_index(drop=True).copy()
df_test = df_to_split[df_to_split['gadget_id'].isin([5,6])].reset_index(drop=True).copy()

print(f"Training data: {df_train.shape}")
print(f"Test data: {df_test.shape}")

### PREDICTION PARAMETERS
w0 = 1
w1 = 8
pos_label = 1

### NEURAL NETWORK
nn = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(10),
    random_state=1,
    max_iter=10000,
    activation='relu',
    tol=0.00001,
)
nn.fit(df_train[X], df_train[y])
df_test['nn'] = nn.predict(df_test[X])

### RANDOM FOREST MODEL
random_forest = RandomForestClassifier(
    min_samples_leaf=7,
    random_state=45,
    n_estimators=50,
    class_weight={0:w0, 1:w1}
)
random_forest.fit(df_train[X], df_train[y])
df_test['random_forest'] = random_forest.predict(df_test[X])

### LOGISTIC REGRESSION MODEL
log_regr = LogisticRegression(class_weight={0:w0, 1:w1})
log_regr.fit(df_train[X], df_train[y])
df_test['log_regr'] = log_regr.predict(df_test[X])

### LINEAR REGRESSION MODEL
lin_regr = Lasso(alpha=0.1, positive=True)
lin_regr.fit(df_train[X], df_train[y])
df_test['lin_regr'] = lin_regr.predict(df_test[X])
df_test['lin_regr'] = np.where(df_test['lin_regr']>=0.5,1,0)

### KNN MODEL
def knn_weights(knn_y):
    return np.where(knn_y==1, w1, w0)
knn = KNeighborsClassifier(weights=knn_weights)
knn.fit(df_train[X], df_train[y])
df_test['knn'] = knn.predict(df_test[X])

### SVM
svm = SVC(
    class_weight={0:w0, 1:w1},
    C=1,
    random_state=42,
    kernel='linear'
)
svm.fit(df_train[X], df_train[y])
df_test['svm'] = svm.predict(df_test[X])

### NAIVE BAYES
bayes = GaussianNB()
bayes.fit(df_train[X], df_train[y])
df_test['bayes'] = bayes.predict(df_test[X])



### PRINT RESULTS
model_summary = []
models = ['random_forest', 'log_regr', 'lin_regr', 'knn', 'nn', 'svm', 'bayes']
for m in models:
    print(f"\n-----------\n{m}")
    try:
        precision = precision_score(df_test['fail_in_1h'], df_test[m], zero_division=0, pos_label=pos_label)
        recall = recall_score(df_test['fail_in_1h'], df_test[m], pos_label=pos_label)
        
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(df_test[m].value_counts())

        model_summary.append({
            'model': m,
            'precision': precision,
            'recall': recall
        })

    except:
        print("Can't calculate score")

#PRINT RESULT DATAFRAME
#print(df_test[['gadget_id', 'measurement_time'] + cols + models].head(5))

#CREATE IMAGE FOR MODEL COMPARISON
df_summary = pd.DataFrame(model_summary)

x = np.arange(len(df_summary['model']))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, df_summary['precision'], width, label='Precision')
rects2 = ax.bar(x + width/2, df_summary['recall'], width, label='Recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Result')
ax.set_title('Precision and Recall by machine learning model')
ax.set_xticks(x)
ax.set_xticklabels(df_summary['model'])
ax.legend()

fig.tight_layout()

plt.savefig('img/results.png')