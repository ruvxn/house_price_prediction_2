import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#import dataset
house_data_original = pd.read_csv('dataset/kc_house_data.csv')

#understand dataste
print(house_data_original.head(10))
print(house_data_original.info())
print(house_data_original.describe())

#feature selection
selected_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
X = house_data_original[selected_features]

#scaling data using MinMaxScaler to scale the data between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#target variable
y = house_data_original['price']

# reshape the target variable as it is a single feature and the model expects a 2D array
y=y.values.reshape(-1,1)
y_scaled=scaler.fit_transform(y)

#split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)

#building the model 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.001)

#compile the model
model.compile(optimizer=optimizer_1, loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=200, batch_size=50, validation_split=0.2)

#evaluate the model
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

#predict the model
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color='r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.title('Model Predictions vs True Values')
plt.show()

#reverse the scaling of the target variable
y_predict_original= scaler.inverse_transform(y_predict)
y_test_original= scaler.inverse_transform(y_test)


plt.plot(y_test_original, y_predict_original, "^", color='r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.title('Model Predictions vs True Values')
plt.show()


RMSE = np.sqrt(tf.keras.losses.mean_squared_error(y_test_original, y_predict_original))
MSE = np.sqrt(mean_squared_error(y_test_original, y_predict_original))
MAE = tf.keras.losses.mean_absolute_error(y_test_original, y_predict_original)
R2 = tf.keras.losses.r2_score(y_test_original, y_predict_original)
adj_R2 = 1 - (1-R2)*(len(y_test_original)-1)/(len(y_test_original)-X_test.shape[1]-1)


print(f"""
Model Performance Metrics:

1. RMSE (Root Mean Squared Error): {RMSE:.4f}
   - Meaning: RMSE measures the standard deviation of the residuals (errors).
   - Expected Value: Should be as low as possible. Closer to 0 means better model performance.

2. MSE (Mean Squared Error): {MSE:.4f}
   - Meaning: MSE is the average of the squared differences between actual and predicted values.
   - Expected Value: Lower values indicate a better fit, but it's sensitive to large errors.

3. MAE (Mean Absolute Error): {MAE:.4f}
   - Meaning: MAE is the average absolute difference between actual and predicted values.
   - Expected Value: Smaller is better, and it's less sensitive to outliers than MSE.

4. R² Score (Coefficient of Determination): {R2:.4f}
   - Meaning: R² represents how well the model explains the variance in the target variable.
   - Expected Value: 
     - R² = 1 → Perfect prediction
     - R² = 0 → Model performs as bad as taking the mean
     - R² < 0 → Model performs worse than a simple mean prediction

5. Adjusted R² Score: {adj_R2:.4f}
   - Meaning: Adjusted R² adjusts the R² value based on the number of predictors.
   - Expected Value: 
     - If adj_R² is close to R², the predictors add real value.
     - If adj_R² is much lower, it suggests overfitting due to too many irrelevant predictors.

""")

#save the model
model.save('house_price_model.h5')
model.save('house_price_model.keras')
print("Model saved as house_price_model.keras/hs")

