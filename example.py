from astropy.io import fits
from classify import classifier
import pandas as pd 
from sklearn.model_selection import train_test_split

#prepare data into proper format to train model, use alldata.fits
data=fits.open('/path/to/alldata.fits')

df = pd.DataFrame(data[1].data)
df['target'] = df['target'].astype(np.int8)

for column in df.columns:
    if df[column].dtype.byteorder == '>':  # Big-endian
        df[column] = df[column].values.byteswap().newbyteorder()


target_column = 'target'
y = df['target']
feature_columns = ['RA', 'DEC', 'VMAG', 'BV_COLOR', 'PORB', 'FLUX', 'FLUX_MAX', 'LII', 'BII', 'VMAG_MIN', 'UB_COLOR', 'PULSE_PERIOD']
X = df[feature_columns]


#split data into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

#train the model using data with features and target
model=classifier.BinaryClassifier(X_train,y_train)
model.train()

# returns confusion matrix plot and accuracy of the model
cm,accuracy=model.evaluate(X_test,y_test)

#classify the new data with necessary parameters
prediction= model.predict(new_X_test)
