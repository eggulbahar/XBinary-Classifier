import numpy as np  
from astropy.io import fits  
from matplotlib import pyplot as plt 
import os  
import pandas as pd 
import seaborn as sns  
# All the ML stuff will come from the scikit-learn package (so do pip install scikit-learn)
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.metrics import classification_report   
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.svm import SVC   
from sklearn.model_selection import train_test_split   
from sklearn.tree import DecisionTreeRegressor   
from sklearn.tree import DecisionTreeClassifier   
from sklearn.model_selection import cross_val_score   
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from astropy import units as u
# Set up directories
home = os.path.expanduser('~')  
workdir = os.path.join(home, "XBinary-Classifier")  
# Read the FITS file w/ the training data
filepath = os.path.join(workdir, 'training.fits')  
with fits.open(filepath) as hdul:   
    data = hdul[1].data  
# Python doesn't like going from fits to dataframe, as it changes the 'endianness' of the data(?)
# Check and convert endianness 
if data.dtype.byteorder == '>':  # Big-endian
    data = data.byteswap().newbyteorder()  # Convert to little-endian
# Create DataFrame
df = pd.DataFrame(data.tolist(), columns=data.names)  
print(df.columns)  # Print the column names
# Convert data types
df['target'] = df['target'].astype(np.int8)   
df['FLUX_MAX'] = df['FLUX_MAX'].astype(np.float64)    
# Compute color ratio (example with BV_COLOR and VMAG)
df['Color_Ratio'] = df['BV_COLOR'] / df['UB_COLOR']
# Compute ratio of VMAG to BV_COLOR
df['VMAG_BV_Ratio'] = df['VMAG'] / df['BV_COLOR']
# Calculaye distance to the galactic center
R0 = 8.34 * u.kpc  # Distance from Sun to GC
df['Galactic_Distance'] = R0 * (
    np.cos(np.radians(df['BII'])) * np.cos(np.radians(df['LII'])) +
    np.sqrt(1 - np.cos(np.radians(df['BII']))**2 * np.cos(np.radians(df['LII']))**2))
# Set up target and feature variables. Drop unnecessary columns.
target_column = 'target'  # In here LMXB are 0, and HMXBs are 1
targets = df[target_column]  
features = df.drop(['NAME', 'target', 'FX', 'FX_MAX', 'PULSE_PER', 'FLUX',  'PULSE_PERIOD', 'RA', 'DEC', 'BV_COLOR','UB_COLOR', 'VMAG', 'VMAG_MIN'], axis=1)  
# Handle missing and inf values
features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=0)   
idxs = features.index               # Get the indices of the remaining rows
targets = targets.iloc[idxs]        # Align target variable with the remaining rows
# Split data into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, random_state=0) 
# Initialize and train the KNN classifier (choosing 4 neighbors)
knn = KNeighborsClassifier(n_neighbors=4)   
KNN_fit = knn.fit(features_train, targets_train)  
# Evaluate the classifier (accuracy)
accuracy = KNN_fit.score(features_test, targets_test)  
print('KNN score: {}\n'.format(accuracy))  
# Perform cross-validation
cv_scores = cross_val_score(knn, features, targets, cv=3) 
print('Cross-validation scores (3-fold):', cv_scores)  
print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))  
# Feature Selection Analysis (Method: Permutation feature importance)
result = permutation_importance(KNN_fit, features_test, targets_test, n_repeats=10, random_state=0)
sorted_idx = result.importances_mean.argsort()
# Plot permutation importances
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=features.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show() 
# Make predictions on the test set and evaluate
predictions = KNN_fit.predict(features_test)  
print(targets_test.shape, predictions.shape) 
# Create and plot the confusion matrix
cm = confusion_matrix(targets_test, predictions) 
# Define class labels
class_labels = ['LMXB', 'HMXB']
# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', annot_kws={"size": 12})
plt.xlabel('Model', fontsize=14)
plt.ylabel('Task', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, fontsize=12)
plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, fontsize=12)
plt.tight_layout() 
plt.show()
