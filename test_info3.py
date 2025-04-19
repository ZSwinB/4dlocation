import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import combinations

# Speed of light (m/s)
c = 299792458

# Receiver positions (x, y, z) - in meters
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

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
        from scipy.optimize import least_squares
        result = least_squares(residuals, initial_guess, method='lm')
        
        return result.x
    except Exception:
        # Return None if estimation fails
        return None

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

def advanced_model_analysis(model, X_test, y_test, all_combinations, df):
    """
    Perform advanced analysis of model performance based on reflection order sum
    
    Args:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: True labels
        all_combinations: List of receiver combinations
        df: Original DataFrame with ray type information
    
    Returns:
        Detailed performance metrics
    """
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Prepare lists to track performance
    total_samples = len(y_test)
    exact_matches = 0
    near_matches_1 = 0  # Matches within ±1 reflection order sum
    near_matches_2 = 0  # Matches within ±2 reflection order sum
    
    # Lists to store sums for plotting
    true_sums = []
    pred_sums = []
    
    # Iterate through test samples
    for i in range(total_samples):
        # Get true reflection orders for selected combination
        true_reflection_orders = np.array([
            int(df.iloc[i]['TOA1_ray_type']), 
            int(df.iloc[i]['TOA2_ray_type']), 
            int(df.iloc[i]['TOA3_ray_type']),
            int(df.iloc[i]['TOA4_ray_type']), 
            int(df.iloc[i]['TOA5_ray_type']), 
            int(df.iloc[i]['TOA6_ray_type'])
        ])
        
        # True combination
        true_indices = all_combinations[y_test[i]]
        true_sum = true_reflection_orders[list(true_indices)].sum()
        
        # Predicted combination
        pred_indices = all_combinations[y_pred[i]]
        pred_sum = true_reflection_orders[list(pred_indices)].sum()
        
        # Store sums for visualization
        true_sums.append(true_sum)
        pred_sums.append(pred_sum)
        
        # Check match types
        if true_sum == pred_sum:
            exact_matches += 1
        elif abs(true_sum - pred_sum) <= 1:
            near_matches_1 += 1
        elif abs(true_sum - pred_sum) <= 2:
            near_matches_2 += 1
    
    # Calculate percentages
    exact_match_percentage = exact_matches / total_samples * 100
    near_match_1_percentage = near_matches_1 / total_samples * 100
    near_match_2_percentage = near_matches_2 / total_samples * 100
    
    # Visualize true vs predicted reflection order sums
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(true_sums, pred_sums, alpha=0.5, 
                c=np.abs(np.array(true_sums) - np.array(pred_sums)), 
                cmap='viridis')
    plt.colorbar(scatter, label='Absolute Difference in Reflection Order Sum')
    plt.title('True vs Predicted Reflection Order Sums')
    plt.xlabel('True Reflection Order Sum')
    plt.ylabel('Predicted Reflection Order Sum')
    
    # Add perfect match line
    min_sum = min(min(true_sums), min(pred_sums))
    max_sum = max(max(true_sums), max(pred_sums))
    plt.plot([min_sum, max_sum], [min_sum, max_sum], 'r--', label='Perfect Match Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reflection_order_sums.png')
    plt.close()
    
    # Detailed performance print
    print("\n### Advanced Model Performance Analysis ###")
    print(f"Exact Match Percentage (Sum): {exact_match_percentage:.2f}%")
    print(f"Near Match (±1 Sum): {near_match_1_percentage:.2f}%")
    print(f"Near Match (±2 Sum): {near_match_2_percentage:.2f}%")
    
    return {
        'exact_match_percentage': exact_match_percentage,
        'near_match_1_percentage': near_match_1_percentage,
        'near_match_2_percentage': near_match_2_percentage,
        'true_sums': true_sums,
        'pred_sums': pred_sums
    }

def modified_main():
    try:
        # Read the data
        excel_path = r"D:\desktop\毕设材料\processed_data.xlsx"
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
        
        # Load the pre-trained model
        with open("toa_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Prepare test data
        features = []
        labels = []
        
        # All possible 3-receiver combinations
        all_combinations = list(combinations(range(6), 3))
        
        # Feature extraction (similar to training process)
        for index, row in df.iterrows():
            try:
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
                    continue
                
                # Extract features
                row_features = extract_features(row, receivers)
                if row_features is None:
                    continue
                
                features.append(row_features)
                labels.append(true_combo_idx)
            
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        # Convert to numpy arrays
        X_test = np.array(features)
        y_test = np.array(labels)
        
        # Run advanced analysis
        performance_metrics = advanced_model_analysis(model, X_test, y_test, all_combinations, df)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    modified_main()