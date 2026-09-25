import os
import glob
import pandas as pd
import numpy as np

def update_master_df_with_stim_amps():
    print("=== Extracting Stimulation Amplitudes & Updating master_df.csv ===")
    
    master_csv = "master_df.csv"
    if not os.path.exists(master_csv):
        print(f"❌ Error: {master_csv} not found.")
        return
        
    mdf = pd.read_csv(master_csv, low_memory=False)
    mdf["Cell_ID"] = mdf["Cell_ID"].astype(str).str.strip()
    
    # Locate all pkl files
    data_dir = "/Users/samgritz/Library/CloudStorage/Box-Box/Milstein-Shared/Sam/GNB1_Paper_Data/Electrophysiology_Experiments/All_Combined_Data"
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        return
        
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} .pkl files in data directory.")
    
    records = []
    
    for f in pkl_files:
        cell_id_raw = f.replace("_processed_data.pkl", "").replace(".pkl", "")
        parts = cell_id_raw.split("_")
        if len(parts) >= 2 and len(parts[0]) == 8:
            mm, dd, yyyy = parts[0][:2], parts[0][2:4], parts[0][4:]
            cell_id = f"{yyyy}{mm}{dd}_{parts[1]}"
        else:
            cell_id = cell_id_raw
            
        try:
            df = pd.read_pickle(os.path.join(data_dir, f))
        except Exception as e:
            continue
            
        if "stimulus_metadata_dict" not in df.columns:
            continue
            
        c1_amps, c2_amps = [], []
        c1_u_amps, c2_u_amps = [], []
        c1_labels, c2_labels = set(), set()
        
        for idx, row in df.iterrows():
            m = row.get("stimulus_metadata_dict", {})
            if isinstance(m, dict):
                c1 = m.get("channel_1_amp", "")
                c2 = m.get("channel_2_amp", "")
                l1 = m.get("channel_1_label", "")
                l2 = m.get("channel_2_label", "")
                isi = m.get("ISI", "")
                
                try:
                    c1_num = float(c1) if str(c1) not in ["", "nan", "None"] else np.nan
                except: c1_num = np.nan
                try:
                    c2_num = float(c2) if str(c2) not in ["", "nan", "None"] else np.nan
                except: c2_num = np.nan
                
                if pd.notna(c1_num) and c1_num > 0:
                    c1_amps.append(c1_num)
                    if str(isi) in ["300", "300.0"]: c1_u_amps.append(c1_num)
                if pd.notna(c2_num) and c2_num > 0:
                    c2_amps.append(c2_num)
                    if str(isi) in ["300", "300.0"]: c2_u_amps.append(c2_num)
                    
                if l1: c1_labels.add(l1)
                if l2: c2_labels.add(l2)

        c1_rep = pd.Series(c1_u_amps).mode()[0] if c1_u_amps else (pd.Series(c1_amps).mode()[0] if c1_amps else np.nan)
        c2_rep = pd.Series(c2_u_amps).mode()[0] if c2_u_amps else (pd.Series(c2_amps).mode()[0] if c2_amps else np.nan)

        records.append({
            "Cell_ID": cell_id,
            "channel_1_amp": c1_rep,
            "channel_2_amp": c2_rep,
            "channel_1_label": ", ".join(c1_labels) if c1_labels else "",
            "channel_2_label": ", ".join(c2_labels) if c2_labels else ""
        })

    df_amps = pd.DataFrame(records)
    
    # Merge extracted amplitudes with master_df
    # Remove old columns if existing to avoid duplicates
    for col in ["channel_1_amp", "channel_2_amp", "channel_1_label", "channel_2_label",
                "Perforant_Stim_Amp", "Schaffer_Stim_Amp", "Stratum_Oriens_Stim_Amp"]:
        if col in mdf.columns:
            mdf.drop(columns=[col], inplace=True)

    mdf = mdf.merge(df_amps, on="Cell_ID", how="left")
    
    # Map pathway-specific amplitudes
    perf_amps = []
    schaffer_amps = []
    oriens_amps = []

    for idx, row in mdf.iterrows():
        pathway_str = str(row.get("Stimulation Pathways", "")).lower()
        l1 = str(row.get("channel_1_label", "")).lower()
        l2 = str(row.get("channel_2_label", "")).lower()
        
        c1_amp = row.get("channel_1_amp", np.nan)
        c2_amp = row.get("channel_2_amp", np.nan)
        
        p_amp = np.nan
        s_amp = np.nan
        o_amp = np.nan
        
        # Check Channel 1
        if "perforant" in pathway_str or "perforant" in l1:
            p_amp = c1_amp
        elif "stratum oriens" in pathway_str or "stratum oriens" in l1 or "oriens" in l1:
            o_amp = c1_amp
        elif "schaffer" in pathway_str or "schaffer" in l1:
            s_amp = c1_amp
            
        # Check Channel 2
        if "schaffer" in pathway_str or "schaffer" in l2:
            s_amp = c2_amp
        elif "perforant" in pathway_str or "perforant" in l2:
            p_amp = c2_amp
        elif "stratum oriens" in pathway_str or "stratum oriens" in l2 or "oriens" in l2:
            o_amp = c2_amp
            
        perf_amps.append(p_amp)
        schaffer_amps.append(s_amp)
        oriens_amps.append(o_amp)

    mdf["Perforant_Stim_Amp"] = perf_amps
    mdf["Schaffer_Stim_Amp"] = schaffer_amps
    mdf["Stratum_Oriens_Stim_Amp"] = oriens_amps
    
    # Save back to master_df.csv
    mdf.to_csv(master_csv, index=False)
    print(f"✓ Updated {master_csv} with channel_1_amp, channel_2_amp, and pathway stim amplitudes.")
    
    # Print summary
    inc = mdf[mdf["Inclusion"].astype(str).str.contains("Yes", case=False, na=False)]
    print("\nSummary of Extracted Stimulation Amplitudes (Included Cells):")
    print(f"  - Channel 1 Stim Amp: {inc['channel_1_amp'].notna().sum()} cells")
    print(f"  - Channel 2 Stim Amp: {inc['channel_2_amp'].notna().sum()} cells")
    print(f"  - Perforant Path Stim Amp: {inc['Perforant_Stim_Amp'].notna().sum()} cells")
    print(f"  - Schaffer Collateral Stim Amp: {inc['Schaffer_Stim_Amp'].notna().sum()} cells")
    print(f"  - Stratum Oriens Stim Amp: {inc['Stratum_Oriens_Stim_Amp'].notna().sum()} cells")

if __name__ == "__main__":
    update_master_df_with_stim_amps()
