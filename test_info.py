import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Speed of light (m/s)
c = 299792458

# Excel file path
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

# Reading Excel data - using precise method
try:
    # Try using openpyxl engine to maintain precision
    df = pd.read_excel(excel_path, header=None, engine='openpyxl')
    print("Original data shape:", df.shape)
    
    # View the first few rows of original data to check precision
    print("First few rows of original data:")
    pd.set_option('display.float_format', '{:.10e}'.format)  # Show more decimal places
    print(df.head(3))
    
    # Manually specify column names
    column_names = ['x', 'y', 'label', 
                   'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                   'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    
    # Ensure column names match DataFrame columns
    if len(column_names) > df.shape[1]:
        column_names = column_names[:df.shape[1]]
    elif len(column_names) < df.shape[1]:
        for i in range(len(column_names), df.shape[1]):
            column_names.append(f'unknown_col{i+1}')
    
    df.columns = column_names
    
    # Display processed data and check precision
    print("\nFirst 5 rows of processed data (maintaining original precision):")
    print(df[['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']].head())
    
    # Verify each column has enough variability (ensure values aren't all the same)
    for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
        unique_values = df[col].nunique()
        print(f"Column {col} has {unique_values} different values")
        if unique_values < 10:
            print("WARNING: This column has low variability, possible reading issue!")
    
except Exception as e:
    print(f"Error reading Excel file: {e}")
    import traceback
    print(traceback.format_exc())
    raise

# Alternative reading method: if above method still has issues, try using xlrd directly
try:
    import xlrd
    print("\nTrying to read Excel file directly with xlrd...")
    
    workbook = xlrd.open_workbook(excel_path)
    sheet = workbook.sheet_by_index(0)
    
    # Check first few rows of data
    print("First few rows read with xlrd:")
    for i in range(min(3, sheet.nrows)):
        row_values = sheet.row_values(i)
        print(f"Row {i+1}: {row_values}")
    
    # If normal reading failed, rebuild DataFrame using xlrd
    if 'TOA1' in df.columns and df['TOA1'].nunique() < 10:
        print("Due to pandas reading inaccuracy, rebuilding DataFrame using xlrd...")
        
        data = []
        for i in range(sheet.nrows):
            row = sheet.row_values(i)
            data.append(row)
        
        df_xlrd = pd.DataFrame(data)
        
        # Name columns for new DataFrame
        if len(column_names) > df_xlrd.shape[1]:
            column_names = column_names[:df_xlrd.shape[1]]
        elif len(column_names) < df_xlrd.shape[1]:
            column_names.extend([f'unknown_col{i+1}' for i in range(len(column_names), df_xlrd.shape[1])])
        
        df_xlrd.columns = column_names
        
        # Check precision of xlrd-read data
        print("\nFirst 5 rows of data read with xlrd:")
        print(df_xlrd[['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']].head())
        
        # Check variability
        for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
            unique_values = df_xlrd[col].nunique()
            print(f"Column {col} read with xlrd has {unique_values} different values")
        
        # Replace original DataFrame with xlrd-read DataFrame
        df = df_xlrd
    
except ImportError:
    print("xlrd not installed, skipping this attempt...")
except Exception as e:
    print(f"Error using xlrd: {e}")
    import traceback
    print(traceback.format_exc())

# Data type conversion - maintain original precision
if 'TOA1' in df.columns:
    # Check if already float type
    numeric_cols = ['x', 'y', 'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']
    for col in numeric_cols:
        if col in df.columns:
            # Check column data type
            if not pd.api.types.is_float_dtype(df[col]):
                print(f"Converting column {col} data type, maintaining original precision")
                # For string columns, ensure scientific notation format is correct
                if pd.api.types.is_string_dtype(df[col]):
                    # Ensure consistent scientific notation format without changing values
                    df[col] = df[col].astype(str).str.replace('E-', 'e-', regex=False)
                    df[col] = df[col].astype(str).str.replace('e-0', 'e-', regex=False)
                # Convert to float64 to maintain precision
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)
    
    reflection_cols = ['label', 'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    for col in reflection_cols:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            print(f"Converting column {col} to integer type")
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')

# Check TOA column variability again to ensure data is correct
for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
    if col in df.columns:
        print(f"\nFirst 20 values of column {col}:")
        print(df[col].head(20).tolist())
        
        # Calculate column statistics
        col_std = df[col].std()
        col_min = df[col].min()
        col_max = df[col].max()
        print(f"{col} statistics: min={col_min:.10e}, max={col_max:.10e}, std_dev={col_std:.10e}")

# Ensure all required columns exist
required_cols = ['x', 'y', 'label', 
                'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']

for col in required_cols:
    if col not in df.columns:
        print(f"WARNING: Missing column '{col}', creating default values")
        df[col] = np.nan

# Physical consistency function: calculate TOAs from emitter to all receivers
def calculate_toas(emitter_pos, receiver_positions):
    """
    Calculate theoretical TOA values from emitter to all receivers
    
    Parameters:
    emitter_pos: Emitter position [x, y, z]
    receiver_positions: Array of receiver positions
    
    Returns:
    Array of TOA values
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
    
    Parameters:
    toa_values: All receivers' TOA values
    receiver_indices: Indices of three receivers to use
    receiver_positions: All receiver positions
    
    Returns:
    Estimated emitter position
    """
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

# Calculate physical consistency score
def calculate_consistency_score(estimated_position, toa_values, receiver_positions):
    """
    Calculate physical consistency score for estimated position
    
    Parameters:
    estimated_position: Estimated emitter position
    toa_values: All receivers' TOA values
    receiver_positions: All receiver positions
    
    Returns:
    Consistency score (negative mean error)
    """
    try:
        expected_toas = calculate_toas(estimated_position, receiver_positions)
        toa_errors = np.abs(expected_toas - toa_values)
        consistency_score = -np.mean(toa_errors)
        return consistency_score
    except Exception as e:
        print(f"Error calculating consistency score: {e}")
        return -np.inf  # Return negative infinity as worst score

# Main function: predict lowest reflection order receivers using physical consistency
def predict_lowest_reflection_receivers(row, receiver_positions):
    """
    Predict which three receivers have lowest reflection order
    
    Parameters:
    row: Data row containing TOA values
    receiver_positions: Receiver positions
    
    Returns:
    Indices of predicted three lowest reflection order receivers
    """
    # Extract TOA values and ensure they are float
    try:
        toa_values = np.array([
            float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
            float(row['TOA4']), float(row['TOA5']), float(row['TOA6'])
        ])
    except (ValueError, TypeError) as e:
        print(f"Error processing row: {row}")
        print(f"Error: {e}")
        # Return default combination if conversion fails
        return (0, 1, 2)
    
    # Check data validity
    if np.any(np.isnan(toa_values)) or np.any(np.isinf(toa_values)):
        print(f"Row contains invalid values: {row}")
        return (0, 1, 2)
    
    # All possible three-receiver combinations
    all_combinations = list(combinations(range(6), 3))
    
    best_score = -np.inf
    best_combination = None
    
    # Evaluate each combination
    for combination in all_combinations:
        try:
            # Estimate emitter position
            estimated_position = estimate_position(toa_values, combination, receiver_positions)
            
            # Ensure estimated position is valid
            if np.any(np.isnan(estimated_position)) or np.any(np.isinf(estimated_position)):
                continue
            
            # Calculate consistency score
            score = calculate_consistency_score(estimated_position, toa_values, receiver_positions)
            
            # Update if better score
            if score > best_score:
                best_score = score
                best_combination = combination
        except Exception as e:
            # Handle calculation errors
            print(f"Error calculating combination {combination}: {e}")
            continue
    
    # If no best combination found, return default
    if best_combination is None:
        print("WARNING: Could not find best combination, using default (0,1,2)")
        return (0, 1, 2)
        
    return best_combination

# Main analysis process
def main_analysis():
    print("Starting analysis...")
    
    # Check data validity
    invalid_rows = df.isnull().any(axis=1).sum()
    if invalid_rows > 0:
        print(f"WARNING: {invalid_rows} rows contain NaN values")
    
    # Prepare for results
    predictions = []
    correct_count = 0
    total_reflection_diff = 0
    errors_count = 0
    processed_count = 0
    
    # Process each row
    for index, row in df.iterrows():
        try:
            # Check if current row has NaN values
            if row.isnull().any():
                missing_cols = row.index[row.isnull()].tolist()
                print(f"Row {index} has missing values: {missing_cols}")
                errors_count += 1
                continue
            
            # Get true ray types (reflection orders)
            true_reflection_orders = np.array([
                int(row['TOA1_ray_type']), int(row['TOA2_ray_type']), int(row['TOA3_ray_type']),
                int(row['TOA4_ray_type']), int(row['TOA5_ray_type']), int(row['TOA6_ray_type'])
            ])
            
            # Predict lowest reflection order receivers
            predicted_indices = predict_lowest_reflection_receivers(row, receivers)
            predictions.append(predicted_indices)
            
            # Get true lowest three reflection order receivers
            true_indices = np.argsort(true_reflection_orders)[:3]
            
            # Check if prediction is correct (considering same order cases)
            correct = True
            
            # Handle cases with same reflection orders
            unique_orders = np.unique(true_reflection_orders)
            if len(unique_orders) < 3:
                # Find receivers with lowest order
                lowest_order = unique_orders[0]
                lowest_indices = np.where(true_reflection_orders == lowest_order)[0]
                
                # Check if predicted receivers are all within lowest order range
                if len(lowest_indices) >= 3:
                    correct = all(idx in lowest_indices for idx in predicted_indices)
                else:
                    # If fewer than 3 lowest order receivers, check if included all lowest ones
                    if len(unique_orders) > 1:
                        second_lowest = unique_orders[1]
                        second_lowest_indices = np.where(true_reflection_orders == second_lowest)[0]
                        
                        # Check if selected all lowest order receivers plus needed from second lowest
                        remaining_needed = 3 - len(lowest_indices)
                        correct = (all(idx in np.concatenate([lowest_indices, second_lowest_indices]) for idx in predicted_indices) and
                                  len(set(predicted_indices) & set(lowest_indices)) == len(lowest_indices))
                    else:
                        # If only one order type, any three receivers are correct
                        correct = True
            else:
                # Simple case: check if predicted three indices match true three lowest indices
                correct = set(predicted_indices) == set(true_indices)
            
            if correct:
                correct_count += 1
            
            # Calculate reflection order sum difference
            predicted_sum = np.sum(true_reflection_orders[list(predicted_indices)])
            true_sum = np.sum(true_reflection_orders[list(true_indices)])
            diff = abs(predicted_sum - true_sum)
            total_reflection_diff += diff
            
            # Track successfully processed rows
            processed_count += 1
            
            # Print progress every 100 rows
            if (index + 1) % 100 == 0 or index == 0:
                print(f"Progress: Processed {index + 1} rows")
                # If first row, print detailed results for debugging
                if index == 0:
                    print(f"  Predicted receivers: {predicted_indices}")
                    print(f"  True lowest order receivers: {true_indices}")
                    print(f"  Prediction correct: {correct}")
                    print(f"  Reflection order difference: {diff}")
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            import traceback
            print(traceback.format_exc())
            errors_count += 1
            continue
    
    # Calculate statistics
    total_rows = len(df)
    valid_rows = processed_count
    accuracy = correct_count / valid_rows if valid_rows > 0 else 0
    avg_reflection_diff = total_reflection_diff / valid_rows if valid_rows > 0 else 0
    
    print(f"\nAnalysis complete! Total {total_rows} rows, successfully processed {valid_rows}, errors {errors_count}")
    print(f"Accuracy (completely correct predictions): {accuracy:.4f} ({correct_count}/{valid_rows})")
    print(f"Average reflection order sum difference: {avg_reflection_diff:.4f}")
    
    # Additional analysis: plot correct vs incorrect predictions
    plt.figure(figsize=(10, 6))
    plt.bar(['Correct Predictions', 'Incorrect Predictions'], [accuracy, 1-accuracy], color=['green', 'red'])
    plt.title('Physical Consistency Method Prediction Accuracy')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    
    for i, v in enumerate([accuracy, 1-accuracy]):
        plt.text(i, v+0.01, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png')
    
    # Additional analysis: receiver selection accuracy
    actual_indices = []
    predicted_indices_flat = []
    
    for index, row in df.iterrows():
        try:
            if row.isnull().any():
                continue
                
            true_reflection_orders = np.array([
                int(row['TOA1_ray_type']), int(row['TOA2_ray_type']), int(row['TOA3_ray_type']),
                int(row['TOA4_ray_type']), int(row['TOA5_ray_type']), int(row['TOA6_ray_type'])
            ])
            true_indices = np.argsort(true_reflection_orders)[:3]
            
            if index < len(predictions):
                for idx in true_indices:
                    actual_indices.append(idx)
                for idx in predictions[index]:
                    predicted_indices_flat.append(idx)
        except:
            continue
    
    # Calculate receiver selection accuracy
    if len(actual_indices) > 0 and len(predicted_indices_flat) > 0:
        receiver_accuracy = []
        for i in range(6):
            actual_count = actual_indices.count(i)
            predicted_count = predicted_indices_flat.count(i)
            if actual_count > 0:
                receiver_accuracy.append((i, predicted_count / actual_count))
            else:
                receiver_accuracy.append((i, 0))
        
        print("\nReceiver selection accuracy:")
        for i, acc in receiver_accuracy:
            print(f"Receiver {i+1}: {acc:.4f}")
    
    return accuracy, avg_reflection_diff, predictions

if __name__ == "__main__":
    accuracy, avg_diff, predictions = main_analysis()