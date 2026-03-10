from analysis_utils import *
import os
import sys
import pandas as pd
import numpy as np

# IMPORT BOX UTILITIES
try:
    import box_utils
except ImportError:
    print("\n❌ Error: box_utils.py not found.")
    print("Please ensure box_utils.py is in the same directory as this script.")
    sys.exit(1)

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

PROJECT_DATA_FOLDER = 'All_Combined_Data'

# ==================================================================================================
# Analyze and Export Intrinsic Properties
# ==================================================================================================

def filter_master_df_general_inclusion(master_df):
    """
    Filters the Master DataFrame to only include rows where
    the 'Inclusion' column starts with 'Yes' (case-insensitive).
    """
    if 'Inclusion' not in master_df.columns:
        print("WARNING: 'Inclusion' column not found in Master DF. Returning full dataframe.")
        return master_df

    if 'Cell_ID' in master_df.columns:
        master_df['Cell_ID'] = master_df['Cell_ID'].astype(str)

    # Filter logic: starts with 'yes' (case-insensitive)
    mask = master_df['Inclusion'].astype(str).str.strip().str.lower().str.startswith('yes')
    
    filtered_df = master_df[mask].copy()
    
    dropped_count = len(master_df) - len(filtered_df)
    print(f"Data Filtering: Kept {len(filtered_df)} cells. Dropped {dropped_count} cells.")
    print(f"   (Criteria: Inclusion starts with 'Yes')")
    
    return filtered_df

if __name__ == "__main__":

    # --- 1. Load Master Dataframe ---
    # Priority 1: Check current directory (Repository Root)
    if os.path.exists('master_df.csv'):
        master_df_path = 'master_df.csv'
        print(f"✓ Found Master DF in current directory.")
        
    # Priority 2: Check parent directory (In case script is in a subfolder)
    elif os.path.exists('../master_df.csv'):
        master_df_path = '../master_df.csv'
        print(f"✓ Found Master DF in parent directory.")
        
    # Priority 3: Fallback to manual input
    else:
        print(f"\n⚠ Could not find 'master_df.csv' in the local repository.")
        while True:
            user_input = input("Please drag and drop 'master_df.csv' here: ").strip()
            clean_path = user_input.replace('"', '').replace("'", "")
            if os.path.exists(clean_path):
                master_df_path = clean_path
                break
            print("❌ File not found. Try again.")

    master_df_raw = pd.read_csv(master_df_path, low_memory=False)

    # --- CRITICAL STEP: FILTER FOR GENERAL INCLUSION ("YES") ---
    # We filter only for "Yes" here so we get Input Res, Vm, Access for ALL valid cells.
    # We will handle Voltage Sag exclusion specifically later.
    master_df = filter_master_df_general_inclusion(master_df_raw)

    # --- 2. Get Data Directory (Box Compatible) ---
    data_path = box_utils.get_data_path(target_folder_name=PROJECT_DATA_FOLDER)
    
    if not data_path:
        print("Exiting...")
        sys.exit(1)

    # =========================================================================
    # STEP 3 — Identify cells with a valid Resting Membrane Potential
    # =========================================================================
    # Vm_rest comes from the master_df spreadsheet column 'Vm rest/start (mV)'.
    # We treat cells with Vm_rest >= -40 mV as excluded (depolarised / bad seal).
    VM_REST_THRESHOLD = -40  # mV

    master_df_copy = master_df.copy()
    master_df_copy['Cell_ID'] = master_df_copy['Cell_ID'].astype(str)

    if 'Vm rest/start (mV)' in master_df_copy.columns:
        master_df_copy['Vm rest/start (mV)'] = pd.to_numeric(
            master_df_copy['Vm rest/start (mV)'], errors='coerce')
    
    # Cells that have a valid (hyperpolarised) Vm_rest
    has_valid_vm = (
        master_df_copy['Vm rest/start (mV)'].notna() &
        (master_df_copy['Vm rest/start (mV)'] < VM_REST_THRESHOLD)
    )
    vm_cell_ids = set(master_df_copy.loc[has_valid_vm, 'Cell_ID'])
    print(f"\n{len(vm_cell_ids)} cells have a valid Vm_rest (< {VM_REST_THRESHOLD} mV) in master_df.")

    # =========================================================================
    # STEP 4 — Calculate Input Resistance from test-pulse sweeps
    # =========================================================================
    # IMPORTANT: We ONLY compute Rin for cells that also have a valid Vm_rest.
    # This guarantees that the two measurements are always from the same cell set.
    #
    # Input Resistance is derived from the -50 pA test pulse at the start of
    # every EPSP_stim (and similar) sweep via calculate_input_resistance_from_test_pulse.
    # get_vm_and_rin_from_test_pulses does this automatically across all sweeps.
    print("\nCalculating Input Resistance from test-pulse sweeps...")

    # Build a filtered master_df containing only cells with valid Vm_rest
    master_df_vm_subset = master_df_copy[master_df_copy['Cell_ID'].isin(vm_cell_ids)].copy()

    rin_data = get_vm_and_rin_from_test_pulses(
        data_dir=data_path,
        master_df=master_df_vm_subset,
        vm_rest_threshold=VM_REST_THRESHOLD,   # also filter inside on a per-sweep basis
        pulse_amp_pA=-50,
        pulse_amp_tolerance_pA=10,
        min_pulse_samples=500
    )

    print(f"  -> Input Resistance extracted for {len(rin_data)} / {len(vm_cell_ids)} cells "
          f"with valid Vm_rest.")

    # =========================================================================
    # STEP 5 — Voltage Sag (still extracted from Intrinsic_cell dicts)
    # =========================================================================
    print("\nExtracting Voltage Sag from Intrinsic_cell records...")
    sag_df = analyze_and_export_intrinsic_properties(
        master_df=master_df,
        data_dir=data_path,
        output_path=None,
        properties_to_extract=['Voltage_sag'],
        vm_rest_threshold=VM_REST_THRESHOLD
    )

    # =========================================================================
    # STEP 6 — Build the combined output DataFrame
    # =========================================================================
    # Start from master_df (Vm_rest + metadata), merge in Rin and Voltage_sag.

    vm_lookup  = dict(zip(master_df_copy['Cell_ID'],
                          master_df_copy['Vm rest/start (mV)']))
    sex_lookup = dict(zip(master_df_copy['Cell_ID'],
                          master_df_copy.get('Sex', pd.Series(dtype=str))))
    geno_lookup = dict(zip(master_df_copy['Cell_ID'],
                           master_df_copy['Genotype']))
    acc_lookup = {}
    if 'Access Resistance (From Whole Cell V-Clamp)' in master_df_copy.columns:
        master_df_copy['Access Resistance (From Whole Cell V-Clamp)'] = pd.to_numeric(
            master_df_copy['Access Resistance (From Whole Cell V-Clamp)'], errors='coerce')
        acc_lookup = dict(zip(master_df_copy['Cell_ID'],
                              master_df_copy['Access Resistance (From Whole Cell V-Clamp)']))

    # Voltage sag lookup keyed by Cell_ID
    sag_lookup = {}
    if sag_df is not None and not sag_df.empty and 'Voltage_sag' in sag_df.columns:
        sag_df['Cell_ID'] = sag_df['Cell_ID'].astype(str)
        sag_lookup = dict(zip(sag_df['Cell_ID'], sag_df['Voltage_sag']))

    rows = []
    # Union of cells: all cells that appear in master_df (has Vm_rest)
    # We report Rin as NaN if the test-pulse derived value wasn't found.
    for cell_id in sorted(vm_cell_ids):
        vm_val  = vm_lookup.get(cell_id, np.nan)
        rin_val = rin_data.get(cell_id, {}).get('Input_Resistance_MOhm', np.nan)
        sag_val = sag_lookup.get(cell_id, np.nan)
        acc_val = acc_lookup.get(cell_id, np.nan)
        rows.append({
            'Cell_ID':    cell_id,
            'Genotype':   geno_lookup.get(cell_id, 'Unknown'),
            'Sex':        sex_lookup.get(cell_id, 'Unknown'),
            'Input_Resistance_MOhm':                  rin_val,
            'Voltage_sag':                            sag_val,
            'Vm rest/start (mV)':                     vm_val,
            'Access Resistance (From Whole Cell V-Clamp)': acc_val,
        })

    intrinsic_df = pd.DataFrame(rows)
    print(f"\nCombined output: {len(intrinsic_df)} cells.")
    print(f"  Rin non-NaN: {intrinsic_df['Input_Resistance_MOhm'].notna().sum()}")
    print(f"  Vm  non-NaN: {intrinsic_df['Vm rest/start (mV)'].notna().sum()}")
    print(f"  Sag non-NaN: {intrinsic_df['Voltage_sag'].notna().sum()}")

    # =========================================================================
    # STEP 7 — Apply Voltage-Sag inclusion criteria
    # =========================================================================
    print("\nApplying strict inclusion criteria for Voltage Sag...")
    inclusion_map = dict(zip(master_df['Cell_ID'].astype(str),
                             master_df['Inclusion'].astype(str)))
    sag_dropped = 0
    for index, row in intrinsic_df.iterrows():
        if 'voltage-sag' not in inclusion_map.get(str(row['Cell_ID']), '').lower():
            if pd.notna(row['Voltage_sag']):
                intrinsic_df.at[index, 'Voltage_sag'] = np.nan
                sag_dropped += 1
    print(f"-> Set Voltage Sag to NaN for {sag_dropped} cells "
          f"(Missing 'Voltage-Sag' in Inclusion column).")

    # =========================================================================
    # STEP 7.5 — Apply Input Resistance exclusion (>400 MOhm)
    # =========================================================================
    print("\nApplying strict exclusion criteria for Input Resistance (> 400 MOhm)...")
    rin_dropped = 0
    for index, row in intrinsic_df.iterrows():
        if pd.notna(row['Input_Resistance_MOhm']) and row['Input_Resistance_MOhm'] > 400:
            intrinsic_df.at[index, 'Input_Resistance_MOhm'] = np.nan
            rin_dropped += 1
    print(f"-> Set Input Resistance to NaN for {rin_dropped} cells "
          f"(Input Resistance > 400 MOhm).")


    # =========================================================================
    # STEP 8 — Save
    # =========================================================================
    final_output_path = 'paper_data/Physiology_Analysis/intrinsic_properties.csv'
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    intrinsic_df.to_csv(final_output_path, index=False)
    print(f"\n✓ Analysis complete. Data exported to: {final_output_path}")