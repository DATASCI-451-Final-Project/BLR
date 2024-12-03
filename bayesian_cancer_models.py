
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

class CancerPredictor:
    def __init__(self):
        self.models = {}
        self.traces = {}
        self.scalers = {}
        self.cancer_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df):
        # Convert categorical variables to numeric
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col != 'MCQ230A':  # Don't encode target variable yet
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Handle special values
        df_processed = df_processed.replace([9999, 99999], np.nan)
        
        # Drop columns with too many missing values
        missing_thresh = 0.5
        cols_to_keep = df_processed.columns[df_processed.isnull().mean() < missing_thresh]
        df_processed = df_processed[cols_to_keep]
        
        # Fill remaining missing values with median
        df_processed = df_processed.fillna(df_processed.median())
        
        return df_processed
    
    def fit(self, X_train, y_train):
        # Preprocess features
        X_processed = self.preprocess_data(X_train)
        self.feature_names = X_processed.columns
        
        # Get unique cancer types
        cancer_types = y_train.unique()
        
        # Train model for each cancer type
        for cancer_type in cancer_types:
            print(f"Training model for cancer type {cancer_type}")
            
            # Create binary target
            y_binary = (y_train == cancer_type).astype(int)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            self.scalers[cancer_type] = scaler
            
            try:
                # Create and train model
                with pm.Model() as model:
                    # Priors
                    beta = pm.Normal('beta', mu=0, sigma=10, shape=X_scaled.shape[1])
                    alpha = pm.Normal('alpha', mu=0, sigma=10)
                    
                    # Linear combination
                    mu = alpha + pm.math.dot(X_scaled, beta)
                    
                    # Likelihood 
                    theta = pm.math.sigmoid(mu)
                    y_obs = pm.Bernoulli('y_obs', p=theta, observed=y_binary)
                    
                    # Sample from posterior
                    trace = pm.sample(1000, 
                                    tune=500,
                                    cores=1,
                                    chains=2,
                                    return_inferencedata=True)
                
                self.models[cancer_type] = model
                self.traces[cancer_type] = trace
           
            except Exception as e:
                print(f"Sampling failed for cancer type {cancer_type}: {e}")
                continue
    
    def predict_proba(self, X_new):
        if not isinstance(X_new, pd.DataFrame):
            raise ValueError("X_new must be a pandas DataFrame")
            
        # Preprocess new data
        X_processed = self.preprocess_data(X_new)
        X_processed = X_processed[self.feature_names]
        
        probabilities = {}
        
        for cancer_type in self.models.keys():
            # Scale features
            X_scaled = self.scalers[cancer_type].transform(X_processed)
            
            # Get posterior predictions
            trace = self.traces[cancer_type]
            beta_samples = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
            alpha_samples = trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
            
            # Calculate probability
            mu = alpha_samples + np.dot(X_scaled, beta_samples)
            prob = 1 / (1 + np.exp(-mu))
            
            probabilities[cancer_type] = prob
            
        return probabilities
    
    def predict_top_k(self, X_new, k=3):
        probabilities = self.predict_proba(X_new)
        
        # Sort cancer types by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k predictions
        return sorted_probs[:k]

# Example usage
def train_models(train_data, test_data):
    predictor = CancerPredictor()
    
    # Prepare training data
    X_train = train_data.drop('MCQ230A', axis=1)
    y_train = train_data['MCQ230A']
    
    # Train models
    predictor.fit(X_train, y_train)
    
    # Make predictions on test data
    X_test = test_data.drop('MCQ230A', axis=1)
    predictions = predictor.predict_top_k(X_test)
    
    return predictor, predictions



def visualize_predictions(predictor, test_data, n_samples=5):
    # Get predictions for first n samples
    results = []
    for i in range(n_samples):
        sample = test_data.iloc[[i]]
        preds = predictor.predict_top_k(sample)
        results.append({
            'Patient': f'Patient {sample.SEQN.values[0]}',
            'Actual': sample.MCQ230A.values[0],
            'Pred_1': preds[0][0],
            'Prob_1': preds[0][1],
            'Pred_2': preds[1][0],
            'Prob_2': preds[1][1],
            'Pred_3': preds[2][0],
            'Prob_3': preds[2][1]
        })
    
    results_df = pd.DataFrame(results)
    
    # Create probability plot
    plt.figure(figsize=(12, 6))
    
    # Plot top 3 predictions for each patient
    x = range(len(results))
    width = 0.25
    
    plt.bar([i - width for i in x], results_df['Prob_1'], width, label='1st Prediction', color='steelblue')
    plt.bar([i for i in x], results_df['Prob_2'], width, label='2nd Prediction', color='lightsteelblue')
    plt.bar([i + width for i in x], results_df['Prob_3'], width, label='3rd Prediction', color='lightgray')
    
    plt.xlabel('Patients')
    plt.ylabel('Probability')
    plt.title('Top 3 Cancer Type Predictions by Patient')
    plt.xticks(x, results_df['Patient'], rotation=45)
    plt.legend()
    
    # Print results table
    print("\nPrediction Results:")
    print(results_df[['Patient', 'Actual', 'Pred_1', 'Prob_1']].to_string())
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load data
    train_path = '/Users/yuhanye/Desktop/451 final/data/cancer_data_train.csv'
    test_path = '/Users/yuhanye/Desktop/451 final/data/cancer_data_test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Train models and get predictions
    predictor, predictions = train_models(train_data, test_data)

    # Visualize results
    visualize_predictions(predictor, test_data)
