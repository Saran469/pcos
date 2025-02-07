import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings("ignore")
pcos_data = pd.read_csv(r"C:\Users\HP\Downloads\diabetes (1).csv")


pcos_data = np.array(pcos_data)

X_pcos = pcos_data[:, :-1]
y_pcos = pcos_data[:, -1]
y_pcos = y_pcos.astype('int')
X_pcos = X_pcos.astype('int')

X_train_pcos, X_test_pcos, y_train_pcos, y_test_pcos = train_test_split(X_pcos, y_pcos, test_size=0.3, random_state=0)


log_reg_pcos = LogisticRegression()

log_reg_pcos.fit(X_train_pcos, y_train_pcos)

# Save the model using pickle
pickle.dump(log_reg_pcos, open('pcos_model.pkl', 'wb'))
