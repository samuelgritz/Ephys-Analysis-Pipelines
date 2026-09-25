import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, ks_2samp

# Global df_stats fallback for get_sig
df_stats = None

def compare_distributions_ks(group1, group2):
    """
    Performs Kolmogorov-Smirnov test for curve comparison.
    Ideal for comparing CDFs (e.g., Sholl analysis distributions).
    """
    # Clean NaNs
    g1 = pd.to_numeric(pd.Series(group1), errors='coerce').dropna()
    g2 = pd.to_numeric(pd.Series(group2), errors='coerce').dropna()
    
    if len(g1) == 0 or len(g2) == 0:
        return {'Test': 'KS Test', 'Statistic': np.nan, 'p': np.nan, 'Significance': 'ns', 'Normality': 'N/A'}

    stat, p_val = ks_2samp(g1, g2)
    
    sig = 'ns'
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'
    
    return {
        'Test': 'KS Test', 
        'Statistic': stat, 
        'p': p_val, 
        'Significance': sig, 
        'Normality': 'N/A'
    }

def get_sig(comp_name, df_custom=None):
    target_df = df_custom if df_custom is not None else df_stats
    if target_df is None: return 'ns', '', '(stats missing)'
    row = target_df[target_df['Comparison'].str.contains(comp_name, regex=False, na=False)]
    if row.empty: return 'ns', '', '(stats missing)'
    sig = row.iloc[0]['Significance']
    p_val = row.iloc[0]['P_Value']
    p_str = f"p={p_val:.4f}"
    if p_val < 0.001: p_str = "p<0.001"
    return sig, p_str, sig

# Local stats helper to ensure Figure 2 runs independently of stats_utils
def compare_groups_mannwhitney(group1, group2):
    """
    Performs Mann-Whitney U test and returns a dictionary compatible with the export format.
    """
    # Clean NaNs
    g1 = pd.to_numeric(pd.Series(group1), errors='coerce').dropna()
    g2 = pd.to_numeric(pd.Series(group2), errors='coerce').dropna()
    
    if len(g1) == 0 or len(g2) == 0:
        return {'Test': 'Mann-Whitney U', 'Statistic': np.nan, 'p': np.nan, 'Significance': 'ns', 'Normality': 'N/A'}

    stat, p_val = mannwhitneyu(g1, g2)
    
    sig = 'ns'
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'
    
    return {
        'Test': 'Mann-Whitney U', 
        'Statistic': stat, 
        'p': p_val, 
        'Significance': sig, 
        'Normality': 'Not tested'
    }

def check_normality(data, alpha=0.05):
    """
    Performs Shapiro-Wilk test for normality.
    Returns: True if Normal (p > alpha), False if Non-Normal.
    """
    clean_data = pd.Series(data).dropna()
    if len(clean_data) < 3: 
        return True # Assume normal if N is too small to test
        
    stat, p = stats.shapiro(clean_data)
    return p > alpha

def compare_two_groups(group1, group2, paired=False, alpha=0.05):
    """
    Runs strictly non-parametric statistical tests.
    
    Logic:
    1. Unpaired: ALWAYS use Mann-Whitney U test.
    2. Paired: ALWAYS use Wilcoxon Signed-Rank test (dropping NaNs pairwise).
       
    Returns: dictionary with Test Name, Statistic, P-Value, and Significance String.
    """
    if paired:
        s1 = pd.Series(group1).reset_index(drop=True)
        s2 = pd.Series(group2).reset_index(drop=True)
        df_pair = pd.DataFrame({'g1': s1, 'g2': s2}).dropna()
        g1 = df_pair['g1'].values
        g2 = df_pair['g2'].values
    else:
        g1 = np.array(pd.Series(group1).dropna())
        g2 = np.array(pd.Series(group2).dropna())
    
    # N check
    if len(g1) < 2 or len(g2) < 2:
        return {'Test': 'Insufficient Data', 'Statistic': np.nan, 'p': 1.0, 'Significance': 'ns', 'Normality': 'N/A'}

    # 1. Check Normality (for documentation purposes only)
    norm1 = check_normality(pd.Series(g1))
    norm2 = check_normality(pd.Series(g2))
    
    test_name = ""
    p_val = 1.0
    stat = 0.0
    
    # 2. Select Test (Strictly Non-Parametric)
    if paired:
        test_name = "Wilcoxon Signed-Rank"
        try:
            stat, p_val = stats.wilcoxon(g1, g2)
        except ValueError:
            # Fallback if differences are zero
            p_val = 1.0
            stat = 0.0
    else:
        # Mann-Whitney U for all unpaired
        test_name = "Mann-Whitney U"
        stat, p_val = stats.mannwhitneyu(g1, g2)

    # 3. Format Significance
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'
    else: sig = 'ns'

    return {
        'Test': test_name,
        'Statistic': stat,
        'p': p_val,
        'Significance': sig,
        'Normality': f"G1:{'Norm' if norm1 else 'Non-Norm'}/G2:{'Norm' if norm2 else 'Non-Norm'}"
    }

def compare_groups_ttest(group1, group2, equal_var=True):
    """
    Performs Student's unpaired two-sample t-test.
    Returns: dictionary with Test Name, Statistic, P-Value, and Significance String.
    """
    g1 = pd.to_numeric(pd.Series(group1), errors='coerce').dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors='coerce').dropna().values

    if len(g1) < 2 or len(g2) < 2:
        return {'Test': "Unpaired t-test", 'Statistic': np.nan, 'p': 1.0, 'Significance': 'ns', 'Normality': 'N/A'}

    stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)

    sig = 'ns'
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'

    df_val = len(g1) + len(g2) - 2

    return {
        'Test': "Unpaired t-test",
        'Statistic': stat,
        'p': p_val,
        'Significance': sig,
        'Normality': f"df={df_val}"
    }

def print_stat_result(fig_name, comparison_name, result_dict):
    """Pretty prints the result to console."""
    print(f"[{fig_name}] {comparison_name}")
    print(f"   > Test: {result_dict['Test']} ({result_dict['Normality']})")
    print(f"   > p-value: {result_dict['p']:.5f}  [{result_dict['Significance']}]")
    print("-" * 60)