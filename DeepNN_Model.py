import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import lime
import lime.lime_tabular
import pickle
import tensorflow as tf
#import onnx
from tensorflow.keras.models import load_model
import json
import joblib 
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import RFE
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import json 
from sklearn.impute import KNNImputer  # Import KNNImputer
###################################################################################################################################################### Step 1: Load the dataset

# Load data and preprocess
data = pd.read_csv('file_without_suffix.csv', low_memory=False)
# Drop unnecessary columns
drop_cols = ['THirpMvAHgw', 'kbevqSdQ70w', 'JCdqUjZZuvx', 'WoPIO7Jd8EL', 
             'QcjaZKRl9D4', 'YSjR80QKKXo', 'FtWNuQmVu7j', 'aOHalAjOIrJ',
             'Bivxg5n4goz', 'j9lDBfNNXlz', 'ZFA17ExK3xY', 'EpvHxcDmxyT', 
             'hcqQi9lsEaj', 'TWFRHSAaoK7', 'fwrDFvuQXJf', 'kvQ0Wz3ZqFr', 
             'enrollment', 'sQ4Z6lEiiq6', 'lYNSA4S86Zc']
data.drop(columns=drop_cols, inplace=True)

# Drop columns with more than 10% missing values
data.dropna(axis=1, thresh=int(data.shape[0] * 0.1), inplace=True)

# Separate numerical and categorical columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical columns
label_encoders = {}
target_column = 'J5kKvyU8mpY'

categorical_columns = [col for col in categorical_columns if col != target_column]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str)) + 1
    # Filter out NaN classes or labels
    filtered_classes = [c for c in le.classes_ if str(c) != 'nan']
    label_encoders[col] = {
        'classes': filtered_classes, 
        'mapping': {v: k for k, v in enumerate(filtered_classes, start=1)}  # Store mapping as {encoded_value: original_value}
    }

# Impute missing values only for numerical columns
imputer = KNNImputer(n_neighbors=10)  # Adjust n_neighbors as needed
data_imputed_numeric = imputer.fit_transform(data[numeric_columns])

# Convert back to DataFrame for numeric columns
data[numeric_columns] = pd.DataFrame(data_imputed_numeric, columns=numeric_columns)

# Impute missing values for categorical columns (encoded)
categorical_imputer = KNNImputer(n_neighbors=10)  # You can adjust n_neighbors
data_imputed_categorical = categorical_imputer.fit_transform(data[categorical_columns])

# Convert back to DataFrame for categorical columns
data[categorical_columns] = pd.DataFrame(data_imputed_categorical, columns=categorical_columns)

# Normalize numeric columns
numeric_scaler = StandardScaler()
data[numeric_columns] = numeric_scaler.fit_transform(data[numeric_columns])

# Save scaler
scaler_data = {
    "mean": numeric_scaler.mean_.tolist(),
    "scale": numeric_scaler.scale_.tolist()
}
with open('scaler.json', 'w') as f:
    json.dump(scaler_data, f)

# Save label encoders

    with open('label_encoders.json', 'w') as f:
        safe_label_encoders = {col: enc for col, enc in label_encoders.items() if enc['classes']}
        json.dump(safe_label_encoders, f)

# Save KNN Imputer (only for nearest neighbors logic)
numeric_imputer_data = {
    "missing_values": imputer.missing_values,  # Default is NaN
    "n_neighbors": imputer.n_neighbors,
    # Note: KNN logic itself isn't serialized, this must be implemented manually
}
with open('numeric_knn_imputer.json', 'w') as f:
    json.dump(numeric_imputer_data, f)

# Save categorical KNN Imputer
categorical_imputer_data = {
    "missing_values": categorical_imputer.missing_values,  # Default is NaN
    "n_neighbors": categorical_imputer.n_neighbors,
    # Note: KNN logic itself isn't serialized, this must be implemented manually
}
with open('categorical_knn_imputer.json', 'w') as f:
    json.dump(categorical_imputer_data, f)

joblib.dump(categorical_imputer, 'categorical_knn_imputer.pkl')    

#########################################################################################################################################################
# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert target column to numerical values
y = y.map({'Yes': 1, 'No': 0})

final_data = pd.concat([X, y], axis=1)
final_data.to_csv('final_processed_data.csv', index=False)


# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy = 0.4, k_neighbors = 5, random_state = 100)
X_balanced, y_balanced = smote.fit_resample(X, y)
# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)


# Check if y_balanced is class labels and convert to one-hot encoding
if y_balanced.ndim == 2 and y_balanced.shape[1] == 1:
    num_classes = np.unique(y_balanced).shape[0]  # Determine the number of classes
    y_balanced = to_categorical(y_balanced, num_classes=num_classes)  # Convert to one-hot encoding

# Now, y_balanced should have the correct shape for model training
print(f'Reshaped y_balanced shape: {y_balanced.shape}')  

# Compute class weights
# Check if y_balanced is one-hot encoded; if so, get the class labels correctly
if y_balanced.ndim > 1:  # If it's one-hot encoded
    y_train_classes = np.argmax(y_balanced, axis=1)
else:  # If it's already in class label format
    y_train_classes = y_balanced
    
# Get unique classes from y_train_classes
unique_classes = np.unique(y_train_classes)
class_labels = unique_classes  # This should include all valid labels

# Compute class weights based on the unique classes found
class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train_classes)

# Create a dictionary mapping class labels to their weights
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
#print(class_weight_dict)  # Optional: You can print the class weights for verification
print(class_weight_dict)
print(type(y_test))
print(y_test.shape)
print("the shape of X_train:", X_balanced.shape)
print("the shape of y_train:", y_balanced.shape)

######################################################################################################################################################
# Step 8: Build the Deep Neural Network (DNN) model
input_shape = (X_balanced.shape[1],)
model = models.Sequential()

# Input layer


model.add(layers.Input(shape=input_shape, name = 'input_layer'))

# Fully connected (Dense) layers
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(32, activation='tanh'))


# Output layer with softmax for classification
model.add(layers.Dense(1, activation='sigmoid'))


# Step 9: Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


# Step 10: Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

def predict_fn(x):
    return model.predict(x)

##############################################################################################################################################
# Step: Extract weights from the first layer
weights = model.layers[0].get_weights()[0]  # Get weights from the first dense layer
biases = model.layers[0].get_weights()[1]   # Get biases from the first dense layer

# Step: Calculate the absolute weight for each feature
feature_weights = np.abs(weights).sum(axis=1)  # Sum absolute weights for each feature
feature_names = categorical_columns + numeric_columns  # Combine categorical and numeric feature names

# Step: Create a DataFrame to see weights
weights_data = pd.DataFrame({'Feature': feature_names, 'Weight': feature_weights})

# Sort the DataFrame based on weights
weights_data = weights_data.sort_values(by='Weight', ascending=False)

# Display the weights associated with each feature
print(weights_data)


model.save('model.h5')
# Save the weights to weights.bin


model.save_weights('weights.weights.h5')


model.load_weights('weights.weights.h5')
weights = model.get_weights()
weights_flattened = [w.flatten() for w in weights]
with open('weights.bin', 'wb') as f:
    for weight in weights:
        weight.tofile(f)

import json

# Step: Extract weights from all layers
weights = model.get_weights()

# Convert the weights to a list of lists for JSON compatibility
weights_json_data = {}
for i, weight in enumerate(weights):
    weights_json_data[f'layer_{i}_weights'] = weight.tolist()  # Assign weights to keys based on layer index

# Save to JSON file
with open('model_weights.json', 'w') as json_file:
    json.dump(weights_json_data, json_file)

print("Model weights successfully saved to 'model_weights.json'.")

model.summary()

############################################################################################################################


##############################################################################################################################
# Step 12: Evaluate the model
from sklearn.metrics import accuracy_score
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
y_pred_prob = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f'AUC: {auc:.4f}')

one_minus_auc = 1 - auc
print(f'1 - AUC: {one_minus_auc:.4f}')

# Step 4: Optionally, plot the ROC curve for each class
n_classes = y_test.shape[1] if y_test.ndim > 1 else len(np.unique(y_test))
print(f'Number of classes: {n_classes}')

# Compute AUC for each class
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f'AUC: {auc:.4f}')

one_minus_auc = 1 - auc
print(f'1 - AUC: {one_minus_auc:.4f}')
fpr = {}
tpr = {}
roc_auc = {}

print(type(n_classes))
print(n_classes)

# Ensure the types and shapes are correct
print(type(y_test), y_test.shape)
print(type(y_pred_prob), y_pred_prob.shape)


# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob)
    roc_auc[i] = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curves for each class
plt.figure(figsize=(10, 7))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Step 5: Confusion matrix and classification report (already present in the original code)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test set (binary classification)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Optional: Print confusion matrix as raw data
print("Confusion Matrix:\n", cm)

##################################################################################################################################################


################################################################################################################################################
# Load a single row (for example, the first row)

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model

# Load your existing model
model = load_model('model.h5')
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Load label encoders from the JSON file
with open('label_encoders.json', 'r') as json_file:
    encoding_data = json.load(json_file)

# Load the original dataset to use for testing
data = pd.read_csv('file_without_suffix.csv', low_memory=False)

# Load KNN Imputer parameters from JSON if needed
with open('numeric_knn_imputer.json', 'r') as json_file:
    imputer_params = json.load(json_file)

with open('categorical_knn_imputer.json', 'r') as json_file:
    categorical_imputer_params = json.load(json_file)
    # Load scaler parameters from JSON
with open('scaler.json', 'r') as f:
    scaler_data = json.load(f)

# Instantiate scaler (should be loaded or defined earlier)
numeric_scaler = StandardScaler()  # Assuming it's defined, else load it similarly to the model
numeric_scaler.mean_ = np.array(scaler_data['mean'])
numeric_scaler.scale_ = np.array(scaler_data['scale'])

# Define numeric and categorical column names
numeric_columns = ['Ghsh3wqVTif', 'xcTT5oXggBZ', 'WBsNDNQUgeX', 'HzhDngURGLk', 
    'vZMCHh6nEBZ', 'A0cMF4wzukz', 'IYvO501ShKB', 'KSzr7m65j5q', 'QtDbhbhXw8w', 
    'jnw3HP0Kehx', 'R8wHHdIp2zv', 'gCQbn6KVtTn', 'IrXoOEno4my', 'BQVLvsEJmSq',
    'YFOzTDRhjkF']  # Replace with actual numeric column names

categorical_columns = ['hDaev1EuehO', 'Aw9p1CCIkqL', 'TFS9P7tu6U6','dtRfCJvzZRF', 
'CxdzmL6vtnx','U4jSUZPF0HH','pDoTShM62yi','PZvOW11mGOq','axDtvPeYL2Y', 
'FklL99yLd3h','FhUzhlhPXqV','sM7PAEYRqEP','FZMwpP1ncnZ','QzfjeqlwN2c',
't1wRW4bpRrj','SoFmSjG4m2N','WTz4HSqoE5E','E0oIYbS2lcV','f0S6DIqAOE5',
't6qq4TXSE7n','pD0tc8UxyGg','vKn3Mq4nqOF','ZjimuF1UNdY','qZKe08ZA2Jl',
'b801bG8cIxt','Nf4Tz0J2vA6','pZgD6CYOa96','pg6UUMn87eM','EWsKHldwJxa',
'TevjEqHRBdC','x7uZB9y0Qey','f02UimVxEc2','LRzaAyb2vGk']  # Replace with actual categorical column names


loaded_imputer = joblib.load('categorical_knn_imputer.pkl')  
# Preprocessing function for a single row
def preprocess_single_row(single_row):
    unseen_label_count = 0

    # Ensure single_row is a DataFrame
    if isinstance(single_row, pd.Series):
        single_row = pd.DataFrame(single_row).T  # Convert Series to DataFrame

    # Step 1: Impute missing values for numeric columns
    numeric_imputer = KNNImputer(n_neighbors=imputer_params['n_neighbors'])

    # Extract numeric data from the single row
    numeric_data = single_row[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Replace NaNs with 0 before imputing
    numeric_data.fillna(0, inplace=True)

    # Check which numeric columns are present in the single row
    present_numeric_columns = numeric_data.columns[numeric_data.notnull().any()].tolist()
    
    # Impute only present numeric columns
    if present_numeric_columns:
        imputed_numeric_data = numeric_imputer.fit_transform(numeric_data[present_numeric_columns])
        imputed_numeric_df = pd.DataFrame(imputed_numeric_data, columns=present_numeric_columns)
        
        # Update single_row with imputed numeric values
        single_row[present_numeric_columns] = imputed_numeric_df

    # Step 2: Impute missing values for categorical columns
    categorical_data = single_row[categorical_columns]

    # Ensure no NaNs in categorical columns
    categorical_data.fillna("Unknown", inplace=True)

    # Check which categorical columns are present in the single row
    present_categorical_columns = [col for col in categorical_columns if col in single_row.index]

    # Impute only present categorical columns using preloaded imputer
    if present_categorical_columns:
        imputed_categorical_data = loaded_imputer.transform(categorical_data[present_categorical_columns])
        imputed_categorical_df = pd.DataFrame(imputed_categorical_data, columns=present_categorical_columns)
        
        # Update single_row with imputed categorical values
        single_row[present_categorical_columns] = imputed_categorical_df

    # Step 3: Encode categorical inputs using label encoders
    categorical_inputs = {}
    for column in categorical_columns:
        if column in encoding_data:
            encoder_info = encoding_data[column]
            mapping = encoder_info['mapping']
            
            # Access the scalar value safely
            column_value = single_row[column].iloc[0] if isinstance(single_row, pd.DataFrame) else single_row[column]
            
            # Map or assign default value
            encoded_value = mapping.get(column_value, 0)  # Default to 0 for unseen labels
            unseen_label_count += 1 if encoded_value == 0 else 0
        else:
            # Default for columns without encoding data
            encoded_value = 0
            unseen_label_count += 1
        
        # Update single_row and store encoded values
        categorical_inputs[column] = encoded_value
        single_row[column] = encoded_value

    # Step 4: Scale numeric inputs
    numeric_inputs = single_row[numeric_columns].values.reshape(1, -1)
    scaled_numeric_inputs = numeric_scaler.transform(numeric_inputs)

    # Replace any remaining NaNs in scaled numeric inputs
    scaled_numeric_inputs = np.nan_to_num(scaled_numeric_inputs, nan=0)

    # Combine encoded categorical data and scaled numeric data
    processed_input = np.hstack((list(categorical_inputs.values()), scaled_numeric_inputs[0]))

    # Final check to ensure no NaNs in the processed input
    processed_input = np.nan_to_num(processed_input, nan=0)
    print(f"Unseen labels encountered: {unseen_label_count}")

    return processed_input

# Load the data used for creating the model and select a single row to predict (e.g., index 11)
single_row = data.iloc[11]  # Change the index if you want a different row

# Preprocess the single row
prepared_input = preprocess_single_row(single_row)
print(prepared_input)

# Check for NaN values in prepared_input
if np.isnan(prepared_input).any():
    print("Warning: prepared_input contains NaN values:", prepared_input)

# Make the prediction
prediction = model.predict(np.array([prepared_input]))  # Ensure it's a 2D array
prepared_input = np.nan_to_num(prepared_input, nan=0)


# Get the predicted class and probability
predicted_class = (prediction > 0.5).astype(int)  # Binary classification
predicted_probability = prediction[0][0]

# Map back to the original label if needed (if you have an inverse mapping)
predicted_label = 'Yes' if predicted_class[0] == 1 else 'No'

# Print the results
print(f"Predicted class: {predicted_class[0]}, Predicted label: {predicted_label}, Probability: {predicted_probability:.4f}")
# Check for NaN values in prediction
if np.isnan(prediction).any():
    print("Warning: Prediction contains NaN values:", prediction)
    # Optionally replace NaNs in the prediction to limit runtime errors
    prediction = np.nan_to_num(prediction, nan=0)
# Example of plotting predicted probabilities (if applicable)
import matplotlib.pyplot as plt

plt.hist(prediction, bins=50, alpha=0.7)
plt.axvline(x=0.5, color='r', linestyle='dashed', linewidth=1)
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()