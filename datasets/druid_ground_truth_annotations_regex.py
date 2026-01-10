import pandas as pd
import re

def extract_entities(claim_text):
    """
    Extract x and y from claims like "{x} is a territory of {y}"
    """
    pattern = r'(.+?)\s+is\s+a\s+territory\s+of\s+(.+)'
    match = re.search(pattern, claim_text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def compare_verdict_with_claim(claim_text, verdict_text):
    """
    Compare verdict {z} with claim's {y} to determine True/False
    """
    x, y = extract_entities(claim_text)
    
    if y is None:
        return None, "Could not parse claim format"
    
    verdict_z = str(verdict_text).strip()
    claim_y = str(y).strip()
    
    # Direct comparison
    if verdict_z.lower() == claim_y.lower():
        return True, f"Verdict '{verdict_z}' matches claim '{claim_y}'"
    else:
        return False, f"Verdict '{verdict_z}' does not match claim '{claim_y}'"

# Example usage with sample data
def process_factchecks(df):
    """
    Process a DataFrame with claims and verdicts, preserving all original columns
    """
    results = []
    
    for idx, row in df.iterrows():
        # Get all original columns from the row
        row_dict = row.to_dict()
        
        # Extract fields needed for comparison
        claim = row['claim']
        verdict = row['factcheck_verdict']
        
        # Compute the boolean verdict and explanation
        is_true, explanation = compare_verdict_with_claim(claim, verdict)
        
        # Add the new computed fields to the original row data
        row_dict['boolean_verdict'] = is_true
        
        results.append(row_dict)
    
    return pd.DataFrame(results)


df = pd.read_csv('druid/borderlines.csv')
print(f"Original columns: {list(df.columns)}")
print(f"Original shape: {df.shape}")

result_df = process_factchecks(df)
print(f"Output columns: {list(result_df.columns)}")
print(f"Output shape: {result_df.shape}")

print("Ground truth True: ", result_df['boolean_verdict'].sum())

result_df.to_csv('druid/borderlines_ground_truth_regex.csv', index=False)
print("\nFirst few rows:")
print(result_df.head())