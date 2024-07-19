from astropy.io import fits
from classify import classifier
import pandas as pd 


data=fits.open('/home/mayhem/Downloads/alldata.fits')

df = pd.DataFrame(data[1].data)
df['target'] = df['target'].astype(np.int8)

for column in df.columns:
    if df[column].dtype.byteorder == '>':  # Big-endian
        df[column] = df[column].values.byteswap().newbyteorder()


target_column = 'target'
y = df['target']
feature_columns = ['RA', 'DEC', 'VMAG', 'BV_COLOR', 'PORB', 'FLUX', 'FLUX_MAX', 'LII', 'BII', 'VMAG_MIN', 'UB_COLOR', 'PULSE_PERIOD']
X = df[feature_columns]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
