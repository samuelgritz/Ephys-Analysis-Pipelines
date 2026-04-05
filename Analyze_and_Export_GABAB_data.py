"""
Analyze and Export Unitary GABAB Data for Figure 7.

Uses E_I_traces_for_plotting.pkl (ISI=300 rows) as the source for
unitary GABAB traces. A data point only exists when BOTH the Gabazine
baseline AND the drug trace are present for the same cell (paired).

This ensures N-counts match the paired reference plots.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from analysis_utils import filter_master_df_by_inclusion, calculate_GABAB_area


def rename_genotype(df):
    df = df.copy()
    if 'Genotype' in df.columns:
        df['Genotype'] = df['Genotype'].replace({'GNB1': 'I80T/+'})
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Master DF -----------------------------------------------------------
    if not os.path.exists('master_df.csv'):
        print("❌ master_df.csv not found."); sys.exit(1)
    master_df = pd.read_csv('master_df.csv', low_memory=False)
    master_df = filter_master_df_by_inclusion(master_df)
    print(f"✓ Master DF loaded: {len(master_df)} included cells.")

    # 2. Load E_I traces pkl ------------------------------------------------
    ei_pkl = 'paper_data/E_I_data/E_I_traces_for_plotting.pkl'
    if not os.path.exists(ei_pkl):
        print(f"❌ {ei_pkl} not found."); sys.exit(1)

    with open(ei_pkl, 'rb') as f:
        raw = pickle.load(f)

    ei_df = pd.DataFrame(raw) if isinstance(raw, dict) else raw
    print(f"✓ E_I traces loaded: {len(ei_df)} rows.")

    # 3. Filter ISI = 300 (unitary) -----------------------------------------
    uni = ei_df[ei_df['ISI'] == 300].copy()
    print(f"  ISI=300 rows: {len(uni)}")

    # drug_col_map: label → column in pkl
    drug_col_map = {
        'ML297': 'Gabazine + ML297_Trace',
        'ETX':   'Gabazine + ETX_Trace',
    }

    output_dir = 'paper_data/gabab_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 4. Build paired delta CSV & trace export dict -------------------------
    # Rule: a data point only exists when a cell has BOTH Gabazine_Trace
    # AND the drug trace present in the SAME row (same cell, same pathway,
    # same ISI). This matches the paired reference plots exactly.

    delta_rows    = []
    # Structure: {cell_id: {Genotype, Traces: {condition_label: {pathway: array}}}}
    traces_export = {}

    print("\n--- Paired N-counts ---")
    for drug_label, drug_col in drug_col_map.items():
        for pathway in ['Perforant', 'Schaffer']:
            sub = uni[(uni['Pathway'] == pathway)].copy()

            # PAIRED: must have both Gabazine AND drug trace in the same row
            paired = sub[sub['Gabazine_Trace'].notna() & sub[drug_col].notna()].copy()

            for geno in ['WT', 'GNB1']:
                g = paired[paired['Genotype'] == geno]
                print(f"  {drug_label} | {pathway} | {geno}: {len(g)} paired cells")

                for _, row in g.iterrows():
                    cell_id = str(row['Cell_ID'])
                    gab_trace  = np.asarray(row['Gabazine_Trace'], dtype=float).flatten()
                    drug_trace = np.asarray(row[drug_col],         dtype=float).flatten()

                    # --- Store traces for overlay plot ---
                    if cell_id not in traces_export:
                        traces_export[cell_id] = {'Genotype': geno, 'Traces': {}}
                    t = traces_export[cell_id]['Traces']
                    if 'Gabazine' not in t:
                        t['Gabazine'] = {}
                    if f'Gabazine + {drug_label}' not in t:
                        t[f'Gabazine + {drug_label}'] = {}
                    t['Gabazine'][pathway]               = gab_trace
                    t[f'Gabazine + {drug_label}'][pathway] = drug_trace

                    # --- Compute paired areas & delta ---
                    pre  = calculate_GABAB_area(gab_trace)
                    post = calculate_GABAB_area(drug_trace)
                    delta = post - pre

                    delta_rows.append({
                        'Cell_ID':          cell_id,
                        'Genotype':         geno,
                        'Pathway':          pathway,
                        'Drug':             drug_label,
                        'Delta_GABAB_Area': delta,
                        'Pre_GABAB_Area':   pre,
                        'Post_GABAB_Area':  post,
                    })

    # 5. Save -----------------------------------------------------------------
    df_delta = pd.DataFrame(delta_rows)
    print(f"\n  Total paired delta rows: {len(df_delta)}")
    print(df_delta.groupby(['Drug', 'Pathway', 'Genotype']).size().reset_index(name='n').to_string())

    delta_path = os.path.join(output_dir, 'GIRK_Unitary_GABAB_Deltas.csv')
    df_delta.to_csv(delta_path, index=False)
    print(f"\n✓ Exported Unitary GABAB Deltas to: {delta_path}")

    traces_path = os.path.join(output_dir, 'Figure7_Unitary_Traces.pkl')
    with open(traces_path, 'wb') as f:
        pickle.dump(traces_export, f)
    n_wt  = sum(1 for v in traces_export.values() if v['Genotype'] == 'WT')
    n_gnb = sum(1 for v in traces_export.values() if v['Genotype'] == 'GNB1')
    print(f"✓ Exported traces to: {traces_path}  (WT={n_wt}, GNB1={n_gnb} cells)")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)