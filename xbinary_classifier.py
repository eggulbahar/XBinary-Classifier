import joblib
import warnings
model = joblib.load('model.pkl')
warnings.filterwarnings('ignore', message='X does not have valid feature names, but KNeighborsClassifier was fitted with feature names')
def xbinary_classifier(source):

    source = [source]
    
    prediction = model.predict(source)
    
    if prediction[0] == 0:
        print('The X-ray Binary source has been classified as low mass')
    else:
        print('The X-ray Binary source has been classified as high mass')