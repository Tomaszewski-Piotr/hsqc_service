import images as images
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pickle
import neptune
import os

token = os.getenv('NEPTUNE_API_TOKEN')
upload = False

if token:
    print('Uploading results')
    neptune.init(project_qualified_name='piotrt/hsqc', api_token=os.getenv('NEPTUNE_API_TOKEN'))
    neptune.create_experiment(name='xgboost')
    upload = True
else:
    print('NEPTUNE_API_TOKEN not specified in the shell, not uploading results')

data_files = images.get_data_files()
print(len(data_files))

#x_all = np.array(shape=(len(data_files), channels, target_height, target_width), dtype=np.float32)
x_all = np.ndarray(shape=(len(data_files), images.target_height*images.target_width*images.channels), dtype=np.uint8)
y_all = np.ndarray(shape=(len(data_files), 5), dtype=np.float32)

i = 0
no_file = len(data_files)
for file in data_files:
    print(i,'/', no_file, ' File:',str(file))
    #extract data scaled down to 224x224
    x_all[i] = np.array(np.ravel(images.preprocess_image(file)))
    #extract required output
    y_all[i] = np.array(images.extract_values(file)).astype(np.float32)
    i+=1

X = pd.DataFrame(data=x_all, index=None, columns=None)
y = pd.DataFrame(data=y_all, index=None, columns=images.value_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

xgb_reg = MultiOutputRegressor(xgb.XGBRegressor(verbosity=3, tree_method='gpu_hist', gpu_id=0))
print(xgb_reg)

print(X_train.shape)
print(y_train.shape)

xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test) # Predictions

y_true = y_test # True values

MSE = mse(y_true, y_pred)

RMSE = np.sqrt(MSE)

R_squared = r2_score(y_true, y_pred)

if upload:
    neptune.log_metric("RMSE", np.round(RMSE, 2))

print("\nRMSE: ", np.round(RMSE, 2))

print()

print("R-Squared: ", np.round(R_squared, 2))

actual = pd.DataFrame(data=y_true, index=None, columns=images.value_names)
predicted = pd.DataFrame(data=y_pred, index=None, columns=images.pred_names)
actual.to_csv('actxgboost.csv')
predicted.to_csv('predxgboost.csv')
pickle.dump(xgb_reg, open("xgboost.dat", "wb"))
if upload:
    neptune.log_artifact('actxgboost.csv')
    neptune.log_artifact('predxgboost.csv')
    neptune.stop()

