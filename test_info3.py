import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import time
import matplotlib.pyplot as plt
#这个混合了机器学习，但是还有待观察。
# Speed of light (m/s)
c = 299792458

# Excel file path - replace with your data file path
excel_path = r"D:\desktop\毕设材料\processed_data.xlsx" 

# Receiver positions (x, y, z) - in meters
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

# Physical consistency function: calculate TOAs from emitter to all receivers
def calculate_toas(emitter_pos, receiver_positions):
    """
    Calculate theoretical TOA values from emitter to all receivers
    """
    # Ensure emitter_pos is a 1D array
    emitter_pos = np.array(emitter_pos).flatten()
    
    # Calculate distance from emitter to each receiver
    distances = []
    for receiver_pos in receiver_positions:
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((receiver_pos - emitter_pos)**2))
        distances.append(distance)
    
    # Convert to TOA
    toas = np.array(distances) / c
    return toas

# Estimate emitter position based on three receivers' TOA
def estimate_position(toa_values, receiver_indices, receiver_positions):
    """
    Estimate emitter position using three receivers' TOA
    """
    try:
        # Select three receivers
        selected_receivers = np.array([receiver_positions[i] for i in receiver_indices])
        selected_toas = np.array([toa_values[i] for i in receiver_indices])
        
        # Initial guess (average of receiver positions)
        initial_guess = np.mean(selected_receivers, axis=0)
        
        # Define residual function
        def residuals(pos):
            calculated_toas = calculate_toas(pos, selected_receivers)
            return calculated_toas - selected_toas
        
        # Solve using least squares
        result = least_squares(residuals, initial_guess, method='lm')
        
        return result.x
    except Exception as e:
        # Return None if estimation fails
        return None

# Calculate physical consistency score
def calculate_consistency_score(estimated_position, toa_values, receiver_positions):
    """
    Calculate physical consistency score for estimated position
    """
    try:
        expected_toas = calculate_toas(estimated_position, receiver_positions)
        toa_errors = np.abs(expected_toas - toa_values)
        consistency_score = -np.mean(toa_errors)
        return consistency_score
    except Exception:
        return -np.inf  # Return negative infinity as worst score

# Extract features for machine learning model
def extract_features(row, receiver_positions):
    """
    Extract features from TOA data for machine learning model
    """
    # Extract TOA values
    try:
        toa_values = np.array([
            float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
            float(row['TOA4']), float(row['TOA5']), float(row['TOA6'])
        ])
    except (TypeError, ValueError) as e:
        # If can't convert to float, print error and return None
        print(f"Error extracting TOA values: {e}")
        return None
    
    features = []
    
    # Feature set 1: TOA values themselves
    features.extend(toa_values)
    
    # Feature set 2: TOA differences (contains physical relationship info)
    toa_diffs = []
    for i in range(6):
        for j in range(i+1, 6):
            toa_diffs.append(abs(toa_values[i] - toa_values[j]))
    features.extend(toa_diffs)
    
    # Feature set 3: Physical consistency scores for each combination
    all_combinations = list(combinations(range(6), 3))
    consistency_scores = []
    position_estimates = {}
    
    # Calculate consistency scores and store position estimates
    for combo_idx, combo in enumerate(all_combinations):
        try:
            position = estimate_position(toa_values, combo, receiver_positions)
            if position is not None:
                position_estimates[combo_idx] = position
                score = calculate_consistency_score(position, toa_values, receiver_positions)
                consistency_scores.append(score)
            else:
                consistency_scores.append(-np.inf)
        except Exception:
            consistency_scores.append(-np.inf)
    
    features.extend(consistency_scores)
    
    # Feature set 4: Rank of each combination based on consistency score
    # Convert scores to ranks (higher score = better rank)
    score_ranks = np.zeros(len(consistency_scores))
    score_indices = np.argsort(consistency_scores)[::-1]  # Sort in descending order
    for rank, idx in enumerate(score_indices):
        score_ranks[idx] = rank
    
    features.extend(score_ranks)
    
    # Feature set 5: Normalized scores and relative differences
    if np.isfinite(consistency_scores).any():
        max_score = max(filter(np.isfinite, consistency_scores))
        min_score = min(filter(np.isfinite, consistency_scores))
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # Normalize scores to 0-1 range
        normalized_scores = [(s - min_score) / score_range if np.isfinite(s) else 0 for s in consistency_scores]
        features.extend(normalized_scores)
        
        # Add relative score differences
        top_score = normalized_scores[score_indices[0]] if len(score_indices) > 0 else 0
        score_diffs = [top_score - s for s in normalized_scores]
        features.extend(score_diffs)
    else:
        # If no valid scores, add zeros
        features.extend([0] * len(consistency_scores) * 2)
    
    return features

# Function to train the TOA model
def train_toa_model(df, receiver_positions, test_size=0.2, model_path="toa_model.pkl"):
    """
    Train a machine learning model to predict the lowest reflection order receivers
    
    Parameters:
    df: DataFrame with TOA and ray type data
    receiver_positions: Array of receiver positions
    test_size: Proportion of data to use for testing
    model_path: Where to save the trained model
    
    Returns:
    Trained model and test data for evaluation
    """
    print("Starting model training process...")
    start_time = time.time()
    
    # Check if data is valid
    required_columns = ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                        'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 
                        'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data is missing required columns: {missing_columns}")
    
    # Prepare all possible combinations of 3 receivers
    all_combinations = list(combinations(range(6), 3))
    
    # Prepare data structures for training
    features = []
    labels = []
    skipped_rows = 0
    total_rows = len(df)
    
    print(f"Extracting features from {total_rows} data samples...")
    
    # Process each row in the dataset
    for index, row in df.iterrows():
        try:
            # Skip rows with NaN values
            if row[required_columns].isnull().any():
                skipped_rows += 1
                continue
            
            # Get true ray types (reflection orders)
            true_reflection_orders = np.array([
                int(row['TOA1_ray_type']), int(row['TOA2_ray_type']), int(row['TOA3_ray_type']),
                int(row['TOA4_ray_type']), int(row['TOA5_ray_type']), int(row['TOA6_ray_type'])
            ])
            
            # Get indices of 3 lowest reflection order receivers
            true_indices = tuple(np.argsort(true_reflection_orders)[:3])
            
            # Find which combination matches the true lowest indices
            true_combo_idx = None
            for i, combo in enumerate(all_combinations):
                if set(combo) == set(true_indices):
                    true_combo_idx = i
                    break
            
            if true_combo_idx is None:
                skipped_rows += 1
                continue
            
            # Extract features
            row_features = extract_features(row, receiver_positions)
            if row_features is None:
                skipped_rows += 1
                continue
            
            features.append(row_features)
            labels.append(true_combo_idx)
            
            # Print progress periodically
            if (index + 1) % 500 == 0 or index == 0 or index == total_rows - 1:
                elapsed_time = time.time() - start_time
                print(f"Processed {index + 1}/{total_rows} rows ({(index + 1)/total_rows*100:.1f}%) - Elapsed: {elapsed_time:.1f}s")
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped_rows += 1
            continue
    
    # Check if we have enough data to train
    if len(features) < 100:
        raise ValueError(f"Not enough valid data for training. Only {len(features)} valid samples found.")
    
    print(f"Feature extraction complete. {len(features)} valid samples, {skipped_rows} skipped.")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    test_accuracy = model.score(X_test, y_test)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Return the model and test data for further evaluation
    return model, X_test, y_test, all_combinations

# Function to evaluate the model in more detail
def evaluate_model(model, X_test, y_test, all_combinations, df, receiver_positions):
    """
    Perform detailed evaluation of the trained model
    """
    print("\nPerforming detailed model evaluation...")
    
    # Basic accuracy
    y_pred = model.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importances = model.feature_importances_
    
    # Visualize feature importance (top 20)
    num_features = len(feature_importances)
    feature_names = []
    
    # Create feature names
    # TOA values
    for i in range(6):
        feature_names.append(f"TOA{i+1}")
    
    # TOA differences
    for i in range(6):
        for j in range(i+1, 6):
            feature_names.append(f"Diff_{i+1}_{j+1}")
    
    # Consistency scores
    for i, combo in enumerate(all_combinations):
        feature_names.append(f"Score_{'+'.join(map(str, combo))}")
    
    # Score ranks
    for i, combo in enumerate(all_combinations):
        feature_names.append(f"Rank_{'+'.join(map(str, combo))}")
    
    # Normalized scores
    for i, combo in enumerate(all_combinations):
        feature_names.append(f"Norm_{'+'.join(map(str, combo))}")
    
    # Score differences
    for i, combo in enumerate(all_combinations):
        feature_names.append(f"ScoreDiff_{'+'.join(map(str, combo))}")
    
    # If we have more features than names, add generic names
    while len(feature_names) < num_features:
        feature_names.append(f"Feature_{len(feature_names)}")
    
    # If we have more names than features, truncate
    feature_names = feature_names[:num_features]
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(20), feature_importances[indices[:20]], align="center")
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    print("Feature importance plot saved to feature_importances.png")
    
    # Compare with pure physical consistency method
    print("\nComparing with pure physical consistency method...")
    
    # Get a subset of test data
    test_indices = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    
    correct_ml = 0
    correct_phys = 0
    
    for idx in test_indices:
        test_features = X_test[idx]
        true_label = y_test[idx]
        
        # ML prediction
        ml_pred = model.predict([test_features])[0]
        
        # Extract consistency scores from features
        num_toa = 6
        num_diffs = num_toa * (num_toa - 1) // 2
        start_idx = num_toa + num_diffs
        end_idx = start_idx + len(all_combinations)
        consistency_scores = test_features[start_idx:end_idx]
        
        # Physical consistency prediction
        phys_pred = np.argmax(consistency_scores)
        
        # Check correctness
        if ml_pred == true_label:
            correct_ml += 1
        if phys_pred == true_label:
            correct_phys += 1
    
    print(f"ML model accuracy on sample: {correct_ml/len(test_indices):.4f}")
    print(f"Physical method accuracy on sample: {correct_phys/len(test_indices):.4f}")
    
    if correct_phys > 0:
        print(f"Relative improvement: {(correct_ml-correct_phys)/correct_phys*100:.2f}%")
    
    return accuracy

# Main function to run the training process
def main():
    try:
        # Read the data
        print(f"Reading data from {excel_path}...")
        df = pd.read_excel(excel_path, header=None, engine='openpyxl')
        
        # Assign column names
        column_names = ['x', 'y', 'label', 
                       'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                       'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 
                       'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
        
        # Adjust column names if needed
        if len(column_names) > df.shape[1]:
            column_names = column_names[:df.shape[1]]
        elif len(column_names) < df.shape[1]:
            for i in range(len(column_names), df.shape[1]):
                column_names.append(f'unknown_col{i+1}')
        
        df.columns = column_names
        
        # Convert data types
        numeric_cols = ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        reflection_cols = ['TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 
                          'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
        for col in reflection_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # Check data quality
        print(f"Data shape: {df.shape}")
        print(f"Missing values: {df[numeric_cols + reflection_cols].isnull().sum().sum()}")
        
        # Train the model
        model, X_test, y_test, all_combinations = train_toa_model(
            df, receivers, test_size=0.2, model_path="toa_model.pkl"
        )
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test, all_combinations, df, receivers)
        
        print("\nTraining and evaluation complete!")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()