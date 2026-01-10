"""
Code to get the annotations from the Druid dataset.
What does evidence_stance represent -> How good is the evidence in supporting the claim? 
Inherently, the claim can be considered as always true, 
How should theta correlate with evidence_stance? 
Make buckets: (1) Refutes (2) Supports (3) Neutral (4) Contradictory 
What does 'refutes' mean? The evidence is opposite to the claim
Expectation of how should theta behave in refutes? 
What does 'theta' represent? How much true - false vector changes when context is added. 




"""

import pandas as pd

def get_annotations(dataset_path):
    """
    Get the annotations from the Druid dataset.
    """
    df = pd.read_csv(dataset_path)
    return df


