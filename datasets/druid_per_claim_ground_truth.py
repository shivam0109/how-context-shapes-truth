import pandas as pd 

def convert_to_bool(verdict):
    if isinstance(verdict, bool):
        return verdict
    elif isinstance(verdict, str):
        verdict_lower = verdict.lower().strip()
        if verdict_lower in ['true', 'half-true', 'half true']:
            return True
        elif verdict_lower == 'false':
            return False
        else:
            raise ValueError(f"Invalid DRUID verdict value: {verdict}")
    else:
        raise ValueError(f"Invalid DRUID verdict type: {type(verdict)}")


def per_claim_ground_truth(df):
    claims_ground_truth = {'claim': [], 'ground_truth': []}
    unique_claims = list(df['claim'].unique())
    for claim in unique_claims:
        df_claim = df[df['claim'] == claim]
        gt = df_claim['verdict_bool'].sum()
        if gt>=df_claim.shape[0]/2:
            claims_ground_truth['claim'].append(claim)
            claims_ground_truth['ground_truth'].append(True)
        else:
            claims_ground_truth['claim'].append(claim)
            claims_ground_truth['ground_truth'].append(False)
    
    print("Total claims: ", len(unique_claims))
    print("Total claims with ground truth: ", len(claims_ground_truth['claim']))
    print("Total ground truth True: ", sum(claims_ground_truth['ground_truth']))
    return pd.DataFrame(claims_ground_truth)


def main(input_path, output_path, column_name='factcheck_verdict'):
    df = pd.read_csv(input_path)
    print("Input shape: ", df.shape)
    df['claim'] = df['claim'].apply(lambda x: x.lower().strip())
    df['verdict_bool'] = df[column_name].apply(convert_to_bool)
    claims_ground_truth = per_claim_ground_truth(df)
    claims_ground_truth.to_csv(output_path, index=False)
    return claims_ground_truth

bl_gt = main('druid/borderlines_ground_truth_regex.csv', 'druid/borderlines_per_claim_ground_truth.csv', column_name='boolean_verdict')
pf_gt = main('druid/politifact.csv', 'druid/politifact_per_claim_ground_truth.csv')
sf1_gt = main('druid/sciencefeedback_cluster1.csv', 'druid/sciencefeedback_cluster1_per_claim_ground_truth.csv')

druid_gt = pd.concat([bl_gt, pf_gt, sf1_gt])
druid_gt.to_csv('druid/druid_per_claim_ground_truth.csv', index=False)

print("Final shape: ", druid_gt.shape)
print("Unique claims: ", len(druid_gt['claim'].unique()))
