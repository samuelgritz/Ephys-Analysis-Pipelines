import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
import ast
import matplotlib.ticker as ticker
import pickle

# ---------------------------------------------------------------------------
# Compatibility shim: pandas ≥1.2 dropped multi-dimensional Series indexing
# that older matplotlib relies on. Patch the three most-used Axes methods to
# silently coerce pandas Series → numpy arrays before they reach matplotlib.
# ---------------------------------------------------------------------------
import matplotlib.axes as _mpl_axes
import functools as _functools

def _coerce(v):
    """If v is a pandas Series, return its numpy values; else return as-is."""
    if isinstance(v, pd.Series):
        return v.to_numpy()
    return v

def _patch_ax_method(method_name):
    original = getattr(_mpl_axes.Axes, method_name)
    @_functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        args = tuple(_coerce(a) for a in args)
        kwargs = {k: _coerce(v) for k, v in kwargs.items()}
        return original(self, *args, **kwargs)
    setattr(_mpl_axes.Axes, method_name, wrapper)

for _m in ('plot', 'errorbar', 'fill_between', 'scatter'):
    _patch_ax_method(_m)
# ---------------------------------------------------------------------------

try:
    from analysis_utils import calculate_AHP_duration, AHP_time_to_peak
except ImportError:
    def calculate_AHP_duration(trace, relative_AHP_trough_idx, AP_threshold_value, sampling_rate):
        dt_ms = 1000 / sampling_rate 
        trace_after_trough = trace[relative_AHP_trough_idx:]
        recovery_indices = np.where(trace_after_trough >= AP_threshold_value)[0]
        if len(recovery_indices) > 0:
            return recovery_indices[0] * dt_ms
        else:
            return np.nan 

    def AHP_time_to_peak(peak_idx, relative_AHP_trough_idx, sampling_rate):
        dt_ms = 1000 / sampling_rate
        return (relative_AHP_trough_idx - peak_idx) * dt_ms

# GLOBAL STYLE & CONFIGURATION
COLORS = {
    'WT': 'black',
    'GNB1': 'red',
    'I80T/+': 'red',
    'WT_Male': 'gray',
    'GNB1_Male': 'lightcoral',
    'I80T/+_Male': 'lightcoral',
    'WT_Female': 'black',
    'GNB1_Female': 'darkred',
    'I80T/+_Female': 'darkred'
}

# Display name mapping: data uses 'GNB1', figures show 'I80T/+'
GENOTYPE_DISPLAY = {'GNB1': 'I80T/+'}

def rename_genotype(df, col='Genotype'):
    """Rename GNB1 → I80T/+ for display purposes. Returns a copy."""
    if df is None or col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].replace(GENOTYPE_DISPLAY)
    return df

FI_CURRENTS_TO_PLOT = [50, 100, 150, 200, 250, 300, 350]
DATA_ROOT = 'paper_data'
OUTPUT_FIG_DIR = 'paper_figures'

def setup_publication_style():
    """Sets matplotlib params for publication-quality figures."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['scatter.marker'] = 'o'

def add_subplot_label(ax, label, x=-0.1, y=1.1, fontsize=12, fontweight='bold'):
    """Adds a bold subplot letter (A, B, C...)"""
    ax.text(x, y, label, transform=ax.transAxes, 
            fontsize=fontsize, fontweight=fontweight, va='top', ha='right')

#Data loading and parsing
def load_data(subfolder, filename):
    path = os.path.join(DATA_ROOT, subfolder, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"⚠ Warning: Data not found at {path}")
        return None

def save_current_fig(fig_name):
    if not os.path.exists(OUTPUT_FIG_DIR):
        os.makedirs(OUTPUT_FIG_DIR)
    path = os.path.join(OUTPUT_FIG_DIR, fig_name)
    # Save in multiple formats
    plt.savefig(path + '.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.savefig(path + '.svg', bbox_inches='tight', format='svg', 
                transparent=False, dpi=300)
    plt.savefig(path + '.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved Figure: {fig_name} (.pdf, .svg, .png)")
    plt.close()

def parse_list_string(list_str):
    """Safely converts a string representation of a list into an actual list."""
    if isinstance(list_str, list):
        return list_str
    try:
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError, TypeError):
        return []

# ==================================================================================================
# FIGURE 4 SIGNIFICANCE MARKERS HELPER
# ==================================================================================================

def annotate_fig4(ax, df_s, metric, pathway_key):
        if df_s is None: return
        row = df_s[(df_s['Metric'] == metric) & (df_s['Pathway'] == pathway_key)]
        if row.empty: return
        p = row.iloc[0]['P_Value']
        ylo, yhi = ax.get_ylim()
        # For positive values bracket above, for negative bracket below baseline
        y_bracket = yhi * 0.9 if yhi > 0 else ylo * 0.9
        draw_significance(ax, 0, 1, p, y_bracket, bracket=True)

def load_figure_5_significance_markers():
    """
    Load Figure 4 significance markers from the FDR-corrected stats file.
    
    Returns a DataFrame with columns:
        Analysis, Pathway, Comparison, Main_Effect_Marker, ISI10_Marker, 
        ISI25_Marker, ISI50_Marker, ISI100_Marker, ISI300_Marker, 
        Main_Effect_p, Interaction_p
    
    Markers:
        '#' = Main effect significant (p < 0.05)
        '*' = Interaction significant AND post-hoc FDR-corrected p < 0.05 at that ISI
    """
    markers_file = 'paper_data/E_I_data/Figure_4_Significance_Markers.csv'
    if os.path.exists(markers_file):
        return pd.read_csv(markers_file)
    else:
        print(f"⚠ Warning: Significance markers file not found: {markers_file}")
        print("  Run Figure_4_All_Stats.R to generate significance markers.")
        return None

def get_figure_5_markers(df_markers, analysis, pathway, comparison):
    """
    Get significance markers for a specific analysis/pathway/comparison.
    
    Args:
        df_markers: DataFrame from load_figure_4_significance_markers()
        analysis: 'Gabazine_Amplitude', 'Gabazine_Supralinearity', or 'E_I_Imbalance'
        pathway: 'Perforant', 'Schaffer', or 'Basal_Stratum_Oriens'
        comparison: e.g., 'WT_vs_GNB1_Gabazine', 'WT_Control_vs_Gabazine', etc.
    
    Returns:
        dict with keys:
            'main_effect': '#' if significant, '' otherwise
            'isi_markers': dict with ISI keys (10, 25, 50, 100, 300) and '*' or ''
            'main_p': main effect p-value
            'interaction_p': interaction p-value
    """
    if df_markers is None:
        return {'main_effect': '', 'isi_markers': {10: '', 25: '', 50: '', 100: '', 300: ''}, 
                'main_p': np.nan, 'interaction_p': np.nan}
    
    # Find matching row
    mask = (df_markers['Analysis'] == analysis) & \
           (df_markers['Pathway'] == pathway) & \
           (df_markers['Comparison'] == comparison)
    
    match = df_markers[mask]
    
    if match.empty:
        return {'main_effect': '', 'isi_markers': {10: '', 25: '', 50: '', 100: '', 300: ''}, 
                'main_p': np.nan, 'interaction_p': np.nan}
    
    row = match.iloc[0]
    
    return {
        'main_effect': row['Main_Effect_Marker'] if pd.notna(row['Main_Effect_Marker']) else '',
        'interaction_marker': row['Interaction_Marker'] if pd.notna(row['Interaction_Marker']) else '',
        'isi_markers': {
            10: row['ISI10_Marker'] if pd.notna(row['ISI10_Marker']) else '',
            25: row['ISI25_Marker'] if pd.notna(row['ISI25_Marker']) else '',
            50: row['ISI50_Marker'] if pd.notna(row['ISI50_Marker']) else '',
            100: row['ISI100_Marker'] if pd.notna(row['ISI100_Marker']) else '',
            300: row['ISI300_Marker'] if pd.notna(row['ISI300_Marker']) else ''
        },
        'main_p': row.get('Main_Effect_p', np.nan),
        'interaction_p': row.get('Interaction_p', np.nan)
    }


#Specific FI data

def get_FI_data(fi_df):
    """
    Processes the raw Firing Rate DataFrame.
    Parses strings, explodes to long format, and calculates Mean/SEM.
    """
    if fi_df is None or fi_df.empty:
        return None, None

    # Use a copy to avoid modifying the input raw DataFrame in place
    temp_df = fi_df.copy()
    
    # Parse columns if they are strings
    temp_df['Currents'] = temp_df['Currents_List'].apply(parse_list_string)
    temp_df['Rates'] = temp_df['Firing_Rates_List'].apply(parse_list_string)

    # Explode to long format and filter currents
    rows = []
    for _, row in temp_df.iterrows():
        currents = row['Currents']
        rates = row['Rates']
        
        # Check for valid lists
        if not isinstance(currents, list) or not isinstance(rates, list):
            continue

        for current, rate in zip(currents, rates):
            # Only include the currents specified in the CONFIGURATION
            if current in FI_CURRENTS_TO_PLOT:
                rows.append({
                    'Cell_ID': row['Cell_ID'],
                    'Genotype': row['Genotype'],
                    'Current': current,
                    'Firing_Rate': rate
                })
    
    fi_df_long = pd.DataFrame(rows)

    if fi_df_long.empty:
        return None, None

    # Calculate Mean Firing Rate
    fi_df_final = fi_df_long.groupby(['Genotype', 'Current'])['Firing_Rate'].agg(
        mean_rate='mean', 
        sem_rate=lambda x: np.std(x, ddof=1) / np.sqrt(len(x.dropna())) if len(x.dropna()) > 1 else 0
    ).reset_index()
    
    return fi_df_final, fi_df_long

def prepare_isi_curve_data(df, max_spikes=15):
    """
    Explodes the ISI_Times_List to calculate Mean +/- SEM ISI for the first N spikes.
    """
    if df is None or df.empty: return None
    
    data = []
    for _, row in df.iterrows():
        # Parse the ISI list if it's a string
        isis = row.get('ISI_Times_List', [])
        isis = parse_list_string(isis)
        
        if not isinstance(isis, list) or len(isis) == 0: continue
        
        # Enumerate ISIs to get Spike Index (1st ISI, 2nd ISI...)
        for i, val in enumerate(isis):
            if i >= max_spikes: break 
            data.append({
                'Genotype': row['Genotype'],
                'Spike_Index': i + 1,
                'ISI_Time': val
            })
            
    long_df = pd.DataFrame(data)
    
    if long_df.empty: return None

    # Calculate Mean and SEM per Genotype per Spike Index
    isi_stats = long_df.groupby(['Genotype', 'Spike_Index'])['ISI_Time'].agg(
        mean_isi='mean',
        sem_isi=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    ).reset_index()
    
    return isi_stats

# PLOTTING FUNCTIONS (Standard & Specialized)

def set_ylim_smart(ax, data, y_col, padding_fraction=0.15, floating_baseline=False):
    """
    Set y-axis limits based on data range with padding.
    - Positive-only data: lower limit = 0 (never below zero)
    - Negative-only data: upper limit = 0 (never above zero)
    - Mixed data: symmetric padding around range
    """
    values = data[y_col].dropna().values
    if len(values) == 0:
        return

    y_min = values.min()
    y_max = values.max()
    y_range = y_max - y_min if y_max != y_min else abs(y_max) or 1

    if y_min >= 0:
        # All positive: anchor at 0, pad top only
        lower_limit = -y_range * 0.02 if floating_baseline else 0
        upper_limit = y_max + y_range * padding_fraction
    elif y_max <= 0:
        # All negative: anchor at 0, pad bottom only
        lower_limit = y_min - y_range * padding_fraction
        upper_limit = y_range * 0.02 if floating_baseline else 0
    else:
        # Mixed: pad both sides
        lower_limit = y_min - y_range * padding_fraction
        upper_limit = y_max + y_range * padding_fraction

    ax.set_ylim(lower_limit, upper_limit)


def apply_clean_yticks(ax, n=5, floating_baseline=False):
    """
    Sets n Y-axis ticks with proper anchoring.
    - Positive data:  first tick = 0, grows upward.   Data never clipped.
    - Negative data:  last  tick = 0, grows downward. Data never clipped.
    - Mixed data:     0 as centre anchor, extends both ways.
    Call AFTER setting ylim.
    """
    import numpy as np

    ymin, ymax = ax.get_ylim()
    if abs(ymax - ymin) < 1e-12:
        return

    tol = abs(ymax - ymin) * 1e-6

    def _nice_step(span, n_intervals):
        """Round span/n_intervals UP to nearest nice number."""
        raw = span / n_intervals
        mag = 10 ** np.floor(np.log10(abs(raw) + 1e-30))
        for c in [1, 2, 2.5, 5, 10]:
            if c >= raw / mag - 1e-9:
                return c * mag
        return 10 * mag

    def _decimals(step):
        return max(0, -int(np.floor(np.log10(abs(step) + 1e-30))))

    if ymin >= -tol:
        # ---- POSITIVE DATA: anchor at 0 ----
        step = _nice_step(ymax, n - 1)
        ticks = np.array([i * step for i in range(n)])
        while ticks[-1] < ymax - tol:          # extend until top tick >= ymax
            ticks = np.append(ticks, ticks[-1] + step)
        ticks = ticks[:n]                       # keep first n (0 … top)

    elif ymax <= tol:
        # ---- NEGATIVE DATA: anchor at 0 (top) ----
        step = _nice_step(abs(ymin), n - 1)
        ticks = np.array([-(n - 1 - i) * step for i in range(n)])
        while ticks[0] > ymin + tol:           # extend until bottom tick <= ymin
            ticks = np.insert(ticks, 0, ticks[0] - step)
        ticks = ticks[-n:]                      # keep last n (bottom … 0)

    else:
        # ---- MIXED DATA: anchor at 0 ----
        step = _nice_step(ymax - ymin, n - 1)
        neg_steps = int(np.ceil(abs(ymin) / step))
        pos_steps = int(np.ceil(ymax      / step))
        ticks = np.array([(-neg_steps + i) * step
                          for i in range(neg_steps + pos_steps + 1)])
        # Guarantee full coverage
        while ticks[0]  > ymin + tol:  ticks = np.insert(ticks, 0, ticks[0] - step)
        while ticks[-1] < ymax - tol:  ticks = np.append(ticks, ticks[-1] + step)

    decimals = _decimals(step)
    ticks = np.round(np.array(ticks, dtype=float), decimals)
    ax.set_yticks(ticks.tolist())
    
    if floating_baseline:
        span = ticks[-1] - ticks[0]
        final_ymin = ticks[0] - span * 0.02 if ymin >= -tol else ticks[0]
        final_ymax = ticks[-1] + span * 0.02 if ymax <= tol else ticks[-1]
        ax.set_ylim(final_ymin, final_ymax)
    else:
        ax.set_ylim(ticks[0], ticks[-1])


def plot_bar_scatter(ax, data, x_col, y_col, hue_col, order=None, bar_width=0.6, unique_col=None,
                     override_n_counts=None, show_scatter=True, ymax=None, ymin=None, floating_baseline=False):
    if order is None: order = sorted(data[x_col].unique())
    max_height = 0

    for i, group in enumerate(order):
        subset = data[data[x_col] == group]
        if subset.empty:
            continue
        
        color = COLORS.get(group, 'gray')
        values = subset[y_col].dropna().values
        if len(values) == 0: continue

        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        
        current_max = mean + sem
        if current_max > max_height: max_height = current_max
        
        ax.bar(i, mean, width=bar_width, color=color, alpha=0.5, label=group, edgecolor='none')
        ax.errorbar(i, mean, yerr=sem, fmt='o', color=color, capsize=1, elinewidth=1, markersize=2)
        
        if show_scatter:
            fixed_x = np.full(len(values), i) 
            ax.scatter(fixed_x, values, color=color, s=2, zorder=3)

    ax.set_xticks(range(len(order)))
    
    # Generate labels with N count
    labels = []
    for g in order:
        # Check for override from stats file
        if override_n_counts and g in override_n_counts:
            n = override_n_counts[g]
        else:
            sub = data[data[x_col] == g].dropna(subset=[y_col])
            if unique_col and unique_col in sub.columns:
                n = sub[unique_col].nunique()
            else:
                n = len(sub)
        labels.append(f"{g}\n(n={n})")
        
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_col.replace('_', ' '))
    ax.grid(False)
    
    # Set smart y-limits
    set_ylim_smart(ax, data, y_col, floating_baseline=floating_baseline)
    # Apply optional hard caps BEFORE tick computation
    cur_min, cur_max = ax.get_ylim()
    if ymin is not None: cur_min = ymin
    if ymax is not None: cur_max = ymax
    ax.set_ylim(cur_min, cur_max)
    apply_clean_yticks(ax, floating_baseline=floating_baseline)
    
    # Force-clamp AFTER apply_clean_yticks (which may have expanded ylim)
    if ymin is not None or ymax is not None:
        final_min = ymin if ymin is not None else ax.get_ylim()[0]
        final_max = ymax if ymax is not None else ax.get_ylim()[1]
        # Filter ticks to only those within range
        ticks = [t for t in ax.get_yticks() if final_min - 1e-9 <= t <= final_max + 1e-9]
        ax.set_ylim(final_min, final_max)
        ax.set_yticks(ticks)
    
    return max_height

def plot_scatter(ax, data, x_col, y_col, hue_col, order=None):
    """Plots individual points and mean +/- SEM, with x-axis labels starting at 1."""
    if order is None: order = sorted(data[x_col].unique())
    max_height = 0

    for i, group in enumerate(order):
        x_pos = i + 1 
        subset = data[data[x_col] == group]
        if subset.empty: continue
        
        color = COLORS.get(group, 'gray')
        values = subset[y_col].dropna().values
        if len(values) == 0: continue

        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        
        current_max = mean + sem
        if current_max > max_height: max_height = current_max
        
        ax.errorbar(x_pos, mean, yerr=sem, fmt='o', color=color, capsize=1, elinewidth=1, markersize=2)
        fixed_x = np.full(len(values), x_pos) 
        ax.scatter(fixed_x, values, color=color, s=2, zorder=3)

    ax.set_xticks(np.arange(len(order)) + 1)
    labels = [f"{g}\n(n={len(data[data[x_col] == g].dropna(subset=[y_col]))})" for g in order]
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_col.replace('_', ' '))
    ax.grid(False)
    ax.set_xlim(0.5, len(order) + 0.5)
    return max_height

def plot_protein_expression(ax, df_protein, metric_type, title, y_label, stats_df=None, panel_id=None, show_scatter=True):
    """
    Plots protein expression bar graph with individual data points using plot_bar_scatter.
    metric_type: 'Absolute' or 'Relative'
    """
    # 1. Reshape Data
    data = []
    
    # Look for columns matching the metric type
    for col in df_protein.columns:
        if metric_type not in col: continue
        
        genotype = None
        if 'WT' in col:
            genotype = 'WT'
        elif 'I80T' in col:
            genotype = 'I80T/+' # Map to I80T/+ for color consistency (Red)
            
        if genotype:
            # Extract value (assuming single row df, take the first value)
            val = df_protein.iloc[0][col]
            data.append({'Genotype': genotype, 'Value': val})
        
    df_long = pd.DataFrame(data)
    
    if df_long.empty:
        plot_trace_placeholder(ax, "No Data Found")
        return

    max_h = plot_bar_scatter(ax, df_long, 'Genotype', 'Value', 'Genotype', order=['WT', 'I80T/+'], show_scatter=show_scatter)
    
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(y_label)
    ax.set_xlabel('') 

    
    # 4. Stats
    if stats_df is not None and panel_id is not None:
       
        comp_str = f"{metric_type} Protein Levels"
        
        # Calculate y_pos for stats
        y_pos_stats = max_h + (max_h * 0.1)
        annotate_from_stats(ax, stats_df, panel_id, comp_str, 0, 1, y_pos_stats)


def plot_trace_placeholder(ax, text="Trace Placeholder"):
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, color='gray', style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# DVC plotting functions
def plot_longitudinal_lines(ax, data, x_col, y_col, hue_col, time_order):
    summary = data.groupby([x_col, hue_col])[y_col].agg(['mean', 'sem', 'count']).reset_index()
    time_map = {label: i for i, label in enumerate(time_order)}
    summary['x_pos'] = summary[x_col].map(time_map)
    
    max_height = 0

    for group in summary[hue_col].unique():
        sub = summary[summary[hue_col] == group].sort_values('x_pos')
        color = COLORS.get(group, 'blue')
        
        x = sub['x_pos'].to_numpy()
        y = sub['mean'].to_numpy()
        e = sub['sem'].to_numpy()
        
        ax.errorbar(x, y, yerr=e, fmt='o', 
                    color=color, capsize=1, markersize=2, elinewidth=0.8, label=group)
        ax.plot(x, y, color=color, linestyle='-', linewidth=0.8)
        
        # Track max
        curr_max = (sub['mean'] + sub['sem']).max()
        if curr_max > max_height: max_height = curr_max

    ax.set_xticks(range(len(time_order)))
    ax.set_xticklabels(time_order)
    ax.set_ylabel(y_col)
    return max_height

def plot_dvc_hourly(ax, stats_df, genotype_col='Genotype'):
    ax.axvspan(0, 6, color='blue', alpha=0.1, lw=0)
    ax.axvspan(17, 23, color='blue', alpha=0.1, lw=0)
    ax.axvspan(6, 17, color='white', alpha=0.1, lw=0)

    for group in ['WT', 'GNB1', 'I80T/+']:
        subset = stats_df[stats_df[genotype_col] == group].sort_values('Hour')
        if subset.empty: continue
        color = COLORS.get(group, 'black')
        hours = subset['Hour'].to_numpy()
        mean = subset['Mean'].to_numpy()
        sem = subset['SEM'].to_numpy()
        ax.plot(hours, mean, color=color, label=group, linewidth=1)
        ax.fill_between(hours, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 17, 23])
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Distance (m)')
    ax.grid(False)

def plot_paired_slope_chart(ax, data, id_col, cat_col, val_col, hue_col, category_order):
    cat1, cat2 = category_order
    pivoted = data.pivot(index=id_col, columns=cat_col, values=val_col)
    meta = data[[id_col, hue_col]].drop_duplicates().set_index(id_col)
    pivoted = pivoted.join(meta)
    
    x1, x2 = 0, 1
    max_height = 0

    for group in pivoted[hue_col].unique():
        subset = pivoted[pivoted[hue_col] == group]
        color = COLORS.get(group, 'black')
        
        for _, row in subset.iterrows():
            if pd.notna(row[cat1]) and pd.notna(row[cat2]):
                ax.plot([x1, x2], [row[cat1], row[cat2]], color=color, alpha=0.3, linewidth=0.8)
                max_height = max(max_height, row[cat1], row[cat2])
        
        m1, m2 = subset[cat1].mean(), subset[cat2].mean()
        s1, s2 = subset[cat1].sem(), subset[cat2].sem()
        
        ax.plot([x1, x2], [m1, m2], color=color, linewidth=3, alpha=1.0)
        ax.errorbar(x1, m1, yerr=s1, fmt='o', color=color, capsize=2, markersize=3)
        ax.errorbar(x2, m2, yerr=s2, fmt='o', color=color, capsize=2, markersize=3)

    ax.set_xticks([x1, x2])
    ax.set_xticklabels(category_order)
    ax.set_xlim(-0.5, 1.5)
    return max_height

#AP AHP plotting examples

# Helper to convert Cell_ID format: 20240104_c1 -> 01042024_c1
def convert_cell_id_format(cell_id):
    parts = cell_id.split('_')
    if len(parts) >= 2 and len(parts[0]) == 8:
        date_part = parts[0]
        cell_part = '_'.join(parts[1:])
        # YYYYMMDD -> MMDDYYYY
        new_date = f'{date_part[4:6]}{date_part[6:8]}{date_part[:4]}'
        return f'{new_date}_{cell_part}'
    return cell_id

# Find ~200pA sweeps from IV_stim
def find_200pA_trace_direct(cell_id_file, folder, target=200, tolerance=50):
    """Find sweep closest to target current from IV_stim or FI experiment"""
    import os
    file_path = os.path.join(folder, f'{cell_id_file}_processed_data.pkl')
    if not os.path.exists(file_path):
        return None, None
    
    try:
        import pandas as pd
        import numpy as np
        data = pd.read_pickle(file_path)
        if 'stim_type' in data.columns:
            iv_sweeps = data[data['stim_type'].isin(['IV_stim', 'Coarse_FI', 'Fine_FI', 'Coarse-FI', 'Dend_FI'])]
        else:
            iv_sweeps = data
            
        best_sweep, best_current, best_diff = None, None, float('inf')
        
        for _, row in iv_sweeps.iterrows():
            stim_cmd = row.get('stim_command', None)
            if stim_cmd is not None and hasattr(stim_cmd, '__len__') and len(stim_cmd) > 0:
                acq = row.get('acquisition_frequency', 20000)
                # Find the maximum current level more robustly instead of hardcoded indices
                cmd_arr = np.array(stim_cmd[0] if isinstance(stim_cmd, list) else stim_cmd, dtype=float)
                current = float(np.max(cmd_arr))
                diff = abs(current - target)
                if diff < best_diff and diff <= tolerance:
                    best_diff = diff
                    best_sweep = row['sweep']
                    best_current = round(current, 0)
        return best_sweep, best_current
    except Exception as e:
        print(f"Error finding 200pA trace for {cell_id_file}: {e}")
        return None, None

# LOOKUP HELPERS 
def _match_cell_id(df, target_id):
    """
    Private helper to match a target_id against a DataFrame's Cell_ID column
    handling common formatting differences (Date formats, Leading Zeros).
    """
    if df is None or df.empty: return pd.DataFrame()
    
    # Identify ID column
    id_col = next((c for c in df.columns if c.lower().replace('_', '') == 'cellid'), None)
    if not id_col: return pd.DataFrame()

    # Normalize column and target
    df_ids = df[id_col].astype(str).str.strip().str.lower()
    target = str(target_id).strip().lower()

    # 1. Exact Match
    match = df[df_ids == target]
    
    # 2. Date Format Swap (MMDDYYYY <-> YYYYMMDD)
    if match.empty:
        parts = target.split('_')
        if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
            date_part = parts[0]
            suffix = '_'.join(parts[1:])
            
            # Convert 03142024 -> 20240314
            if date_part[4:].startswith('20'): # MMDDYYYY
                mm, dd, yyyy = date_part[0:2], date_part[2:4], date_part[4:8]
                iso_id = f"{yyyy}{mm}{dd}_{suffix}"
                match = df[df_ids == iso_id]
            # Convert 20240314 -> 03142024
            elif date_part.startswith('20'): # YYYYMMDD
                yyyy, mm, dd = date_part[0:4], date_part[4:6], date_part[6:8]
                us_id = f"{mm}{dd}{yyyy}_{suffix}"
                match = df[df_ids == us_id]

    return match

def get_value_from_master(master_df, cell_id, column_name):
    """Retrieves metadata (like Rheobase Sweep) from master_df."""
    match = _match_cell_id(master_df, cell_id)
    
    if not match.empty:
        # fuzzy column match
        target_col_lower = column_name.lower().replace('_', ' ')
        actual_col = next((c for c in master_df.columns if c.lower().replace('_', ' ') == target_col_lower), None)
        
        if actual_col:
            return match.iloc[0][actual_col]
        else:
            print(f"  ⚠ Column '{column_name}' not found in Master DF.")
    return None

def get_value_from_analysis_df(analysis_df, cell_id, column_name):
    """Retrieves calculated metrics (like AP_threshold) from analysis CSVs."""
    match = _match_cell_id(analysis_df, cell_id)
    
    if not match.empty:
        if column_name in match.columns:
            return match.iloc[0][column_name]
    return None

def get_sweep_index_from_master(master_df, cell_id):
    """Wrapper for getting Rheobase Sweep."""
    # Try common variations
    for col in ['Rheobase Sweep', 'Rheobase_Sweep']:
        val = get_value_from_master(master_df, cell_id, col)
        if val is not None and pd.notna(val):
            return int(float(val))
    return None

def calculate_AHP_duration(trace, relative_AHP_trough_idx, AP_threshold_value, sampling_rate):
    dt_ms = 1000 / sampling_rate 
    trace_after_trough = trace[relative_AHP_trough_idx:]
    recovery_indices = np.where(trace_after_trough >= AP_threshold_value)[0]
    if len(recovery_indices) > 0:
        return recovery_indices[0] * dt_ms
    else:
        return np.nan 

def AHP_time_to_peak(peak_idx, relative_AHP_trough_idx, sampling_rate):
    dt_ms = 1000 / sampling_rate
    return (relative_AHP_trough_idx - peak_idx) * dt_ms

def find_file_for_cell(data_dir, target_cell_id):
    """Scans data_dir RECURSIVELY for a .pkl file matching the target_cell_id."""
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return None
    target_clean = str(target_cell_id).lower().strip()
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.pkl'): continue
            f_lower = f.lower()
            if target_clean in f_lower:
                return os.path.join(root, f)
            if target_clean.startswith('0') and target_clean[1:] in f_lower:
                return os.path.join(root, f)
    return None

def plot_example_AHP_components(ax, trace, sampling_rate=20000, cell_name="Example", analysis_df=None):
    """
    Visualizes AHP components. 
    Now accepts 'analysis_df' (e.g., df_ap_ahp) to look up the threshold.
    """
    dt_sec = 1 / sampling_rate
    time_ms = np.arange(len(trace)) * (dt_sec * 1000)
    
    # --- 1. Determine Threshold ---
    AP_threshold = None
    AP_threshold_idx = None
    
    # Try looking up from the Analysis DataFrame (df_ap_ahp)
    if analysis_df is not None:
        looked_up_thresh = get_value_from_analysis_df(analysis_df, cell_name, 'AP_threshold')
        if looked_up_thresh is not None and not pd.isna(looked_up_thresh):
            AP_threshold = float(looked_up_thresh)
            # Find index
            crossings = np.where(trace >= AP_threshold)[0]
            if len(crossings) > 0: 
                AP_threshold_idx = crossings[0]

    # Fallback to calculation
    if AP_threshold is None or AP_threshold_idx is None:
        derivative = np.diff(trace)
        AP_threshold_idx = np.argmax(derivative)
        AP_threshold = trace[AP_threshold_idx]
    
    # --- 2. Peaks & Troughs ---
    peaks, _ = find_peaks(trace)
    valid_peaks = peaks[peaks > AP_threshold_idx]
    
    if len(valid_peaks) == 0: 
        plot_trace_placeholder(ax, "No Spike Found")
        return
        
    peak_idx = valid_peaks[0]
    min_idx_local = np.argmin(trace[peak_idx:])
    AHP_trough_idx = peak_idx + min_idx_local
    
    # Calculate duration
    # (Ensure calculate_AHP_duration is defined or imported above)
    AHP_duration_ms = calculate_AHP_duration(trace, AHP_trough_idx, AP_threshold, sampling_rate)
    
    duration_samples = int(50 / 1000 * sampling_rate)
    end_idx = min(len(trace) - 1, AHP_trough_idx + duration_samples)

    # --- 3. Plotting ---
    ax.plot(time_ms, trace, color='black', linewidth=1, alpha=0.8)

    if AHP_trough_idx < len(time_ms) and end_idx < len(time_ms):
        ax.fill_between(time_ms[AHP_trough_idx:end_idx + 1], AP_threshold,
                       trace[AHP_trough_idx:end_idx + 1], color='green', alpha=0.3, label='Decay Area')
    
    recovery_idx = AHP_trough_idx + int(AHP_duration_ms / 1000 / dt_sec) if not np.isnan(AHP_duration_ms) else None
    if recovery_idx is not None and recovery_idx < len(time_ms):
        ax.plot([time_ms[AHP_trough_idx], time_ms[recovery_idx]], [AP_threshold, AP_threshold], color='purple', ls=':', lw=2, label='Duration')
    
    ax.axhline(y=AP_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Threshold')
    
    ax.set_title(f'AHP Decay Example')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

def plot_example_rheobase_and_sweeps(ax, data_dir, master_df=None, target_cell_id='03142024_c2', sweep_idx=None, analysis_df=None, show_values=True, show_annotations=True, color='black'):
    file_path = find_file_for_cell(data_dir, target_cell_id)
    if not file_path:
        plot_trace_placeholder(ax, f"File not found: {target_cell_id}")
        return

    if sweep_idx is None:
        sweep_idx = get_sweep_index_from_master(master_df, target_cell_id)
    
    if sweep_idx is None:
        plot_trace_placeholder(ax, f"Sweep Index Not Found")
        return

    try:
        data_df = pd.read_pickle(file_path)
        if 'sweep' not in data_df.columns:
             print(f"Error: 'sweep' column not found in pickle file for {target_cell_id}")
             plot_trace_placeholder(ax, "Sweep Col Missing")
             return

        if sweep_idx >= len(data_df): 
            print(f"Error: Sweep index {sweep_idx} is out of bounds (Max: {len(data_df)-1})")
            plot_trace_placeholder(ax, "Sweep Idx OOB")
            return
            
        rheobase_trace = data_df['sweep'][sweep_idx].copy()
        sampling_rate = 20000  # 20 kHz
        dt_ms = 1000 / sampling_rate  # time step in ms
        time = np.arange(len(rheobase_trace)) * dt_ms
        
        # --- 1. Determine Threshold from CSV ---
        threshold_voltage = None
        ap_thresh_idx = None
        
        # Try looking up from the Analysis DataFrame (analysis_df = df_ap_ahp)
        if analysis_df is not None:
            looked_up_thresh = get_value_from_analysis_df(analysis_df, target_cell_id, 'AP_threshold')
            if looked_up_thresh is not None and not pd.isna(looked_up_thresh):
                threshold_voltage = float(looked_up_thresh)
                # Find index where trace crosses threshold
                crossings = np.where(rheobase_trace >= threshold_voltage)[0]
                if len(crossings) > 0: 
                    ap_thresh_idx = crossings[0]
        
        # Fallback to calculation if CSV lookup failed
        if threshold_voltage is None or ap_thresh_idx is None:
            derivative = np.diff(rheobase_trace)
            ap_thresh_idx = np.argmax(derivative)
            threshold_voltage = rheobase_trace[ap_thresh_idx]
        
        # Find peak
        peaks, _ = find_peaks(rheobase_trace)
        valid_peaks = peaks[peaks > ap_thresh_idx]
        
        if len(valid_peaks) == 0:
            # No AP found, plot simple trace
            ax.plot(time, rheobase_trace, color=color, linewidth=1)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Voltage (mV)')
            ax.set_xlim(300, 900)
            return
            
        peak_idx = valid_peaks[0]
        peak_voltage = rheobase_trace[peak_idx]
        
        # Find AHP trough
        min_idx_local = np.argmin(rheobase_trace[peak_idx:peak_idx+int(50*sampling_rate/1000)])
        ahp_trough_idx = peak_idx + min_idx_local
        ahp_trough_voltage = rheobase_trace[ahp_trough_idx]
        
        # Calculate AP size and AHP amplitude
        ap_size = peak_voltage - threshold_voltage
        ahp_amplitude = threshold_voltage - ahp_trough_voltage
        
        # Zoom in ULTRA EXTREME: show from -1ms before threshold to +20ms after (for maximum closeup)
        zoom_start_idx = max(0, ap_thresh_idx - int(1 * sampling_rate / 1000))
        zoom_end_idx = min(len(rheobase_trace), ap_thresh_idx + int(20 * sampling_rate / 1000))
        
        zoomed_trace = rheobase_trace[zoom_start_idx:zoom_end_idx]
        zoomed_time = time[zoom_start_idx:zoom_end_idx]
        
        # Adjust indices for zoomed view
        peak_idx_zoom = peak_idx - zoom_start_idx
        ahp_trough_idx_zoom = ahp_trough_idx - zoom_start_idx
        ap_thresh_idx_zoom = ap_thresh_idx - zoom_start_idx
        
        # Plot the main trace
        ax.plot(zoomed_time, zoomed_trace, color=color, linewidth=1)
        
        # Add AHP decay shading (from trough to threshold recovery)
        # Find decay duration
        decay_duration_samples = int(50 / 1000 * sampling_rate)  # 50ms max
        decay_end_idx = min(len(rheobase_trace) - 1, ahp_trough_idx + decay_duration_samples)
        
        if show_annotations:
            if decay_end_idx > zoom_start_idx and ahp_trough_idx < zoom_end_idx:
                # Adjust for zoom
                decay_start_zoom = max(0, ahp_trough_idx - zoom_start_idx)
                decay_end_zoom = min(len(zoomed_trace), decay_end_idx - zoom_start_idx)
                
                # Fill between threshold and trace
                ax.fill_between(
                    zoomed_time[decay_start_zoom:decay_end_zoom],
                    threshold_voltage,
                    zoomed_trace[decay_start_zoom:decay_end_zoom],
                    color='lightgreen', alpha=0.3, label='AHP Decay Area'
                )
            
            # Add horizontal reference lines and annotations
            # 1. Threshold line
            ax.axhline(y=threshold_voltage, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            thresh_label = 'AP Threshold' if not show_values else f'AP Threshold: {threshold_voltage:.1f} mV'
            ax.text(zoomed_time[0] + 1, threshold_voltage + 1.5, thresh_label, 
                    fontsize=8, va='bottom', ha='left', color='gray', fontweight='bold')
            
            # 2. Peak marker
            ax.plot(zoomed_time[peak_idx_zoom], peak_voltage, 'ro', markersize=3)
            ax.text(zoomed_time[peak_idx_zoom] + 1, peak_voltage, 'Peak', 
                    fontsize=8, va='bottom', ha='left', color='red')
            
            # 3. AP Size (vertical line from threshold to peak)
            mid_time = zoomed_time[peak_idx_zoom] - 2
            ax.plot([mid_time, mid_time], [threshold_voltage, peak_voltage], 'blue', linewidth=1)
            ax.plot([mid_time-0.5, mid_time+0.5], [threshold_voltage, threshold_voltage], 'blue', linewidth=1)
            ax.plot([mid_time-0.5, mid_time+0.5], [peak_voltage, peak_voltage], 'blue', linewidth=1)
            ap_size_label = 'AP Size' if not show_values else f'AP Size\n{ap_size:.1f} mV'
            ax.text(mid_time - 1, (threshold_voltage + peak_voltage)/2, ap_size_label, 
                    fontsize=7, va='center', ha='right', color='blue')
            
            # 4. AHP Amplitude (vertical line from threshold to trough)
            if ahp_trough_idx_zoom < len(zoomed_time):
                mid_time_ahp = zoomed_time[ahp_trough_idx_zoom] + 3
                ax.plot([mid_time_ahp, mid_time_ahp], [ahp_trough_voltage, threshold_voltage], 
                        'purple', linewidth=1)
                ax.plot([mid_time_ahp-0.5, mid_time_ahp+0.5], [threshold_voltage, threshold_voltage], 'purple', linewidth=1)
                ax.plot([mid_time_ahp-0.5, mid_time_ahp+0.5], [ahp_trough_voltage, ahp_trough_voltage], 'purple', linewidth=1)
                ahp_amp_label = 'AHP Amp' if not show_values else f'AHP Amp\n{ahp_amplitude:.1f} mV'
                ax.text(mid_time_ahp + 1, (threshold_voltage + ahp_trough_voltage)/2, 
                        ahp_amp_label, 
                        fontsize=7, va='center', ha='left', color='purple')
            
            # 5. AP Halfwidth (horizontal line at 50% of AP size)
            halfwidth_voltage = threshold_voltage + (ap_size / 2)
            # Find where trace crosses halfwidth voltage
            rising_crossings = np.where(zoomed_trace[:peak_idx_zoom] >= halfwidth_voltage)[0]
            if len(rising_crossings) > 0:
                hw_start_idx = rising_crossings[0]
                falling_crossings = np.where(zoomed_trace[peak_idx_zoom:] <= halfwidth_voltage)[0]
                if len(falling_crossings) > 0:
                    hw_end_idx = peak_idx_zoom + falling_crossings[0]
                    hw_start_time = zoomed_time[hw_start_idx]
                    hw_end_time = zoomed_time[hw_end_idx]
                    
                    # Draw horizontal line
                    ax.plot([hw_start_time, hw_end_time], [halfwidth_voltage, halfwidth_voltage], 
                           'orange', linewidth=1, alpha=0.9)
                    ax.plot([hw_start_time, hw_start_time], [halfwidth_voltage-1, halfwidth_voltage+1], 'orange', linewidth=1)
                    ax.plot([hw_end_time, hw_end_time], [halfwidth_voltage-1, halfwidth_voltage+1], 'orange', linewidth=1)
                    ax.text((hw_start_time + hw_end_time)/2, halfwidth_voltage + 3, 'Halfwidth',
                           fontsize=8, va='bottom', ha='center', color='orange', fontweight='bold')

        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Voltage (mV)', fontsize=10)
        
    except Exception as e:
        print(f"Error plotting trace: {e}")
        plot_trace_placeholder(ax, f"Error: {e}")

def plot_voltage_sag_example(ax, data_dir, target_cell_id='03142024_c2', master_df=None):
    """Plot voltage sag example trace with annotations showing calculation."""
    file_path = find_file_for_cell(data_dir, target_cell_id)
    if not file_path:
        plot_trace_placeholder(ax, f"File not found: {target_cell_id}")
        return
    
    try:
        data_df = pd.read_pickle(file_path)
        sag_data = data_df[data_df['stim_type'] == 'Voltage_sag']
        if sag_data.empty:
            plot_trace_placeholder(ax, "No Voltage_sag experiment found")
            return
        
        trace = sag_data.iloc[0]['sweep']
        sampling_rate = 20000
        dt_ms = 1000 / sampling_rate
        time = np.arange(len(trace)) * dt_ms
        
        # Correct timing from YAML: step_start=350ms, duration=500ms
        start_time_ms, end_time_ms = 350, 850
        start_idx = int(start_time_ms * sampling_rate / 1000)
        end_idx = int(end_time_ms * sampling_rate / 1000)
        baseline_start_idx = start_idx - int(50 * sampling_rate / 1000)
        
        baseline_voltage = np.mean(trace[baseline_start_idx:start_idx])
        min_voltage = np.min(trace[start_idx:end_idx])
        steady_state_voltage = np.mean(trace[end_idx - int(50 * sampling_rate / 1000):end_idx])
        sag_ratio = abs((steady_state_voltage - min_voltage) / (baseline_voltage - min_voltage)) * 100
        
        # Plot trace
        ax.plot(time, trace, 'k-', linewidth=1)
        
        # Add subtle shaded regions
        ax.axvspan(baseline_start_idx*dt_ms, start_time_ms, alpha=0.1, color='gray', label='Baseline')
        ax.axvspan(start_time_ms, end_time_ms, alpha=0.05, color='blue')
        
        # Horizontal reference lines
        ax.axhline(y=baseline_voltage, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=min_voltage, color='blue', linestyle=':', linewidth=1, alpha=0.8)
        ax.axhline(y=steady_state_voltage, color='green', linestyle='--', linewidth=1, alpha=0.6)
        
        # Annotations with arrows pointing to features
        ax.annotate('Baseline', xy=(start_time_ms-20, baseline_voltage), 
                   xytext=(250, baseline_voltage+2), fontsize=8, color='gray', fontweight='bold')
        ax.annotate('Sag Trough', xy=(500, min_voltage), 
                   xytext=(550, min_voltage-3), fontsize=8, color='blue', fontweight='bold')
        ax.annotate('Steady-state', xy=(end_time_ms-50, steady_state_voltage), 
                   xytext=(end_time_ms-200, steady_state_voltage+2), fontsize=8, color='green', fontweight='bold')
        
        ax.set_ylabel('mV', fontsize=10)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_title('Voltage Sag', fontsize=11, fontweight='bold', loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Show 200ms to 750ms (wider window, less sliced)
        ax.set_xlim(200, 950)
        
        # Set Y-limits explicitly to make it taller (more vertical space)
        y_min = min(min_voltage, steady_state_voltage) - 5
        y_max = baseline_voltage + 3
        ax.set_ylim(y_min, y_max)
        
        # Add scale bar (150ms, 10mV)
        add_scale_bar(ax, 150, 10, x_pos=0.8, y_pos=0.15)
        
    except Exception as e:
        print(f"Error plotting voltage sag: {e}")
        plot_trace_placeholder(ax, f"Error: {e}")

def plot_input_resistance_example(ax, data_dir, target_cell_id='03142024_c2', master_df=None):
    """Plot input resistance example trace with annotations showing calculation."""
    file_path = find_file_for_cell(data_dir, target_cell_id)
    if not file_path:
        plot_trace_placeholder(ax, f"File not found: {target_cell_id}")
        return
    
    try:
        data_df = pd.read_pickle(file_path)
        # Look for voltage_sag experiment which contains the test pulse
        sag_data = data_df[data_df['stim_type'] == 'Voltage_sag']
        
        if sag_data.empty:
            plot_trace_placeholder(ax, "No Voltage_sag experiment found")
            return
        
        target_trace = sag_data.iloc[0]['sweep']
        current_amp = -50  # Test pulse amplitude from YAML
        
        sampling_rate = 20000
        dt_ms = 1000 / sampling_rate
        time = np.arange(len(target_trace)) * dt_ms
        
        # Correct timing from YAML: test pulse 50-150ms
        start_time_ms, end_time_ms = 50, 150
        start_idx = int(start_time_ms * sampling_rate / 1000)
        end_idx = int(end_time_ms * sampling_rate / 1000)
        baseline_start_idx = int(20 * sampling_rate / 1000)
        baseline_end_idx = start_idx
        
        vm_baseline = np.mean(target_trace[baseline_start_idx:baseline_end_idx])
        vm_steady = np.mean(target_trace[end_idx - int(20 * sampling_rate / 1000):end_idx])
        delta_v = vm_steady - vm_baseline
        input_resistance = abs(delta_v / current_amp) * 1000
        
        # Plot trace
        ax.plot(time, target_trace, 'k-', linewidth=1)
        
        # Add shaded regions
        ax.axvspan(baseline_start_idx*dt_ms, start_time_ms, alpha=0.1, color='gray')
        ax.axvspan(start_time_ms, end_time_ms, alpha=0.1, color='lightblue')
        
        # Horizontal reference lines
        ax.axhline(y=vm_baseline, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=vm_steady, color='purple', linestyle='--', linewidth=1, alpha=0.6)
        
        # Vertical bracket showing ΔV
        bracket_x = 180
        ax.annotate('', xy=(bracket_x, vm_steady), xytext=(bracket_x, vm_baseline),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(bracket_x + 5, (vm_baseline + vm_steady)/2, f'ΔV\n{delta_v:.1f}mV',
                fontsize=7, va='center', color='red', fontweight='bold')
        
        # Annotations
        ax.annotate('Baseline', xy=(30, vm_baseline), 
                   xytext=(30, vm_baseline+1.5), fontsize=8, color='gray', fontweight='bold')
        ax.annotate('Steady-state', xy=(130, vm_steady), 
                   xytext=(130, vm_steady-2), fontsize=8, color='purple', fontweight='bold')
        
        ax.set_ylabel('mV', fontsize=10)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_title('Input Resistance', fontsize=11, fontweight='bold', loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Show 0ms to 250ms (wider window)
        ax.set_xlim(0, 200)
        
        # Set Y-limits explicitly to make it taller
        y_min = min(vm_steady, vm_baseline) - 3
        y_max = max(vm_steady, vm_baseline) + 2
        ax.set_ylim(y_min, y_max)
        
        # Add scale bar (50ms, 5mV)
        add_scale_bar(ax, 50, 5, x_pos=0.8, y_pos=0.15)
        
    except Exception as e:
        print(f"Error plotting input resistance: {e}")
        plot_trace_placeholder(ax, f"Error: {e}")

def generate_figure_2_example_plots(data_dir, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_trace_placeholder(axes[0], "Call from main script with master_df")
    plot_trace_placeholder(axes[1], "Call from main script with master_df")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved example figure to {output_path}")
    plt.close()

#Plot Sholl Data
def plot_sholl_data(ax, df, genotype, dendrite_type, color):
        """Plot Sholl analysis for a specific genotype and dendrite type"""
        # Filter for specific dendrite type
        # Negative radius = Basal dendrites
        # Positive radius = Apical dendrites
        if dendrite_type == 'Basal':
            subset = df[df['Radius'] < 0].copy()
            # Use absolute values of radius for plotting distance from soma
            subset['Radius'] = subset['Radius'].abs()
        else:  # Apical
            subset = df[df['Radius'] > 0].copy()
        
        if len(subset) == 0:
            return
        
        # Calculate mean and SEM for each radius
        grouped = subset.groupby('Radius')['Inters.'].agg(['mean', 'sem']).reset_index()
        
        # Calculate proper SEM (accounting for sample size at each radius)
        counts = subset.groupby('Radius')['Inters.'].count().reset_index()
        counts.columns = ['Radius', 'n']
        grouped = grouped.merge(counts, on='Radius')
        grouped['sem'] = grouped['sem'] / np.sqrt(grouped['n'])
        
        # Plot with error bars
        ax.errorbar(
            grouped['Radius'], 
            grouped['mean'], 
            yerr=grouped['sem'],
            fmt='o-',
            color=color,
            capsize=1,
            label=f'{genotype} (n={len(df["Cell_ID"].unique())})',
            alpha=0.8,
            linewidth=0.5,
            markersize=1
        )

#E:I plotting
def select_and_plot_example(ax, genotype, isi, label):
    
    """Select a representative example and plot both pathways"""
    # Filter for this condition
    # Robustly match genotype even if already renamed in df
    genotype_targets = [genotype, 'GNB1', 'I80T/+']
    condition = df_traces[(df_traces['Genotype'].isin(genotype_targets)) & 
                        (df_traces['ISI'] == isi)]
    
    if len(condition) == 0:
        geno_label = GENOTYPE_DISPLAY.get(genotype, genotype)
        ax.text(0.5, 0.5, f'No data for {geno_label} {isi}ms ISI', 
            ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Get unique cells for this condition
    cells = condition['Cell_ID'].unique()
    if len(cells) == 0:
        ax.text(0.5, 0.5, f'No cells for {genotype} {isi}ms ISI',
            ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Select first cell (can be made more sophisticated)
    selected_cell = cells[0]
    
    # Get traces for both channels
    cell_data = condition[condition['Cell_ID'] == selected_cell]
    schaffer = cell_data[cell_data['Channel'] == 'channel_1']
    perforant = cell_data[cell_data['Channel'] == 'channel_2']
    
    # Plot both pathways side by side
    if not schaffer.empty and schaffer.iloc[0]['Control_Trace'] is not None:
        schaffer_trace = schaffer.iloc[0]['Control_Trace']
        time_schaffer = np.arange(len(schaffer_trace)) / 20  # Assuming 20kHz sampling
        ax.plot(time_schaffer, schaffer_trace, 'k-', linewidth=1, label='Schaffer')
    
    if not perforant.empty and perforant.iloc[0]['Control_Trace'] is not None:
        perforant_trace = perforant.iloc[0]['Control_Trace']
        time_perforant = np.arange(len(perforant_trace)) / 20  # Assuming 20kHz sampling
        # Offset perforant path trace for visibility
        ax.plot(time_perforant + time_schaffer[-1] + 50, perforant_trace, 'b-', linewidth=1, label='Perforant')
    
    # Remove axes as requested
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add label
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', fontweight='bold')

# ==================================================================================================
# STATS ANNOTATION HELPER (Integration)
# ==================================================================================================

def draw_significance(ax, x1, x2, p_val, y_pos, bracket=True):
    """Draws the bracket and star."""
    if p_val < 0.001: symbol = '***'
    elif p_val < 0.01: symbol = '**'
    elif p_val < 0.05: symbol = '*'
    else: return # Don't draw ns

    if bracket:
        bar_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=0.8, c='black')
        text_y = y_pos + bar_height
    else:
        text_y = y_pos

    ax.text((x1 + x2) * 0.5, text_y, symbol, ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

    # Automatically expand ylim if the star is too high
    current_ylim = ax.get_ylim()
    if text_y > current_ylim[1] * 0.9:
        ax.set_ylim(current_ylim[0], text_y * 1.15)

def annotate_with_sig_markers(ax, markers_df, analysis, pathway, comparison, x_coords):
    """
    Draws statistical significance markers.
    Bracket string '*' for main effect.
    Bracket string '#' for interaction effect.
    Also includes asterisk for post-hoc at specific data points.
    """
    if markers_df is None: return
    
    # Pathway naming map for CSV matching
    path_map = {'ECIII (Perforant)': 'Perforant', 'Perforant': 'Perforant',
                'CA3 Apical (Schaffer)': 'Schaffer', 'Schaffer': 'Schaffer',
                'CA3 Basal (S. Oriens)': 'Basal_Stratum_Oriens', 'Basal_Stratum_Oriens': 'Basal_Stratum_Oriens',
                'CA3 Basal': 'Basal_Stratum_Oriens'}

    path_csv = path_map.get(pathway, pathway)
    
    match = markers_df[
        (markers_df['Analysis'] == analysis) & 
        (markers_df['Pathway'] == path_csv) & 
        (markers_df['Comparison'] == comparison)
    ]
    
    if match.empty: 
        return
    
    row = match.iloc[0]
    
    main_p = row.get('Main_Effect_p', np.nan)
    inter_p = row.get('Interaction_p', np.nan)
    
    main_sig = pd.notna(main_p) and float(main_p) < 0.05
    inter_sig = pd.notna(inter_p) and float(inter_p) < 0.05

    y_min, y_max_data = ax.get_ylim()
    y_range = y_max_data - y_min
    
    did_draw_extra = False
    x_min, x_max = min(x_coords), max(x_coords)
    isi_to_index = {300: 0, 100: 1, 50: 2, 25: 3, 10: 4}

    # 1. Main Effect and Interaction Brackets
    # Use symbols: '*' for main effect, '#' for interaction
    symbols = []
    if main_sig:
        symbols.append('*')
    if inter_sig:
        symbols.append('#')
        
    if symbols:
        symbol_str = " ".join(symbols)
        bracket_y = y_max_data + y_range * 0.15
        ax.plot([x_min, x_max], [bracket_y, bracket_y], 'k-', linewidth=0.8)
        ax.plot([x_min, x_min], [bracket_y, bracket_y - y_range * 0.02], 'k-', linewidth=0.8)
        ax.plot([x_max, x_max], [bracket_y, bracket_y - y_range * 0.02], 'k-', linewidth=0.8)
        ax.text((x_min + x_max)/2, bracket_y + y_range * 0.02, symbol_str, 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        did_draw_extra = True

    # 2. Post-hoc Markers (*) - Only if interaction is significant
    if inter_sig:
        for isi, idx in isi_to_index.items():
            if idx in x_coords:
                col = f'ISI{isi}_Marker'
                marker = row.get(col, '')
                if pd.notna(marker) and '*' in str(marker):
                    # Draw star a bit above the data point range
                    # Should be below the bracket if bracket exists
                    y_pos = y_max_data + y_range * 0.02
                    ax.text(idx, y_pos, '*', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold', color='black')
                    did_draw_extra = True

    if did_draw_extra:
        # Increase ylim to accommodate markers
        ax.set_ylim(y_min, y_max_data + y_range * 0.35)

def plot_ei_trace_summary(ax, df_traces, pathway_name, isi, trace_col, title, label):
    """
    Plots mean +/- SEM traces for WT and GNB1 from the E_I traces dataframe.
    """
    add_subplot_label(ax, label)
    ax.set_title(title, fontweight='bold', fontsize=8)
    ax.axis('off')
    
    if df_traces is None or df_traces.empty: return
    
    # Pathway naming map for CSV matching
    path_map = {'ECIII (Perforant)': 'Perforant', 'Perforant': 'Perforant',
                'CA3 Apical (Schaffer)': 'Schaffer', 'Schaffer': 'Schaffer',
                'CA3 Basal (S. Oriens)': 'Basal_Stratum_Oriens', 'Basal_Stratum_Oriens': 'Basal_Stratum_Oriens',
                'CA3 Basal': 'Basal_Stratum_Oriens'}

    path_csv = path_map.get(pathway_name, pathway_name)
    
    subset = df_traces[(df_traces['Pathway'] == path_csv) & (df_traces['ISI'] == isi)]
    
    if subset.empty:
        print(f"  Warning: No traces found for {pathway_name} ISI {isi}")
        return

    for geno, color in zip(['WT', 'I80T/+'], ['black', 'red']):
        geno_traces = subset[subset['Genotype'] == geno][trace_col].dropna().values
        if len(geno_traces) == 0: continue
        
        # Ensure all traces are numpy arrays and find min length
        valid_traces = [np.array(t) for t in geno_traces if isinstance(t, (np.ndarray, list))]
        if not valid_traces: continue
        
        min_len = min(len(t) for t in valid_traces)
        valid_traces = np.array([t[:min_len] for t in valid_traces])
        
        mean_trace = np.mean(valid_traces, axis=0)
        sem_trace = np.std(valid_traces, axis=0) / np.sqrt(len(valid_traces))
        time = np.arange(min_len) / 20 # 20 kHz
        
        ax.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.3, edgecolor='none')
        ax.plot(time, mean_trace, color=color, linewidth=1, label=f'{geno} (n={len(valid_traces)})')
    
    add_scale_bar(ax, 50, 1, x_pos=0.85, y_pos=0.15)
    ax.legend(frameon=False, fontsize=7, loc='upper right')

def plot_bar_comparison_df(ax, df_metrics, pathway_name, metric_col, ylabel, label, scale=1.0):
    """
    Plots a bar comparison for WT vs GNB1 from a metrics dataframe.
    """
    add_subplot_label(ax, label)
    
    # Pathway naming map 
    path_map = {'ECIII (Perforant)': 'Perforant', 'Perforant': 'Perforant',
                'CA3 Apical (Schaffer)': 'Schaffer', 'Schaffer': 'Schaffer',
                'CA3 Basal (S. Oriens)': 'Basal_Stratum_Oriens', 'Basal_Stratum_Oriens': 'Basal_Stratum_Oriens',
                'CA3 Basal': 'Basal_Stratum_Oriens'}

    path_csv = path_map.get(pathway_name, pathway_name)
    subset = df_metrics[df_metrics['Pathway'] == path_csv]
    
    if subset.empty: return
    
    # Scale if needed (e.g. mV*s -> mV*ms)
    subset = subset.copy()
    subset.loc[:, metric_col] = subset[metric_col] * scale
    
    # plot_bar_scatter(ax, data, x_col, y_col, hue_col, order=None, ...)
    plot_bar_scatter(ax, subset, 'Genotype', metric_col, None, 
                    order=['WT', 'I80T/+'], unique_col='Cell_ID')
    
    ax.set_ylabel(ylabel, fontsize=7)

def annotate_from_stats(ax, stats_df, panel_id, comparison_substring, x1, x2, y_pos, bracket=True):
    """
    Finds significance marker in stats_df and draws it on ax from x1 to x2 at y_pos.
    Filters by Figure_Panel == panel_id AND Comparison containing comparison_substring.
    """
    if stats_df is None or stats_df.empty: return
    
    # 1. Filter by Panel ID (e.g. 'Fig 1B', 'Fig 2A')
    subset = stats_df[stats_df['Figure_Panel'] == panel_id]
    if subset.empty: return
    
    # 2. Filter by Comparison string (e.g. 'Weight', 'Input Resistance')
    row = subset[subset['Comparison'].str.contains(comparison_substring, case=False, na=False)]
    if row.empty: return
    
    sig = str(row.iloc[0]['Significance'])
    if sig == 'ns' or not sig: return
    
    # 3. Draw
    if bracket:
        h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        ax.plot([x1, x1, x2, x2], [y_pos-h, y_pos, y_pos, y_pos-h], lw=1, color='black')
        ax.text((x1+x2)*0.5, y_pos, sig, ha='center', va='bottom', fontsize=8)
    else:
        ax.text((x1+x2)*0.5, y_pos, sig, ha='center', va='bottom', fontsize=8)

def get_safe_y(data_series, buffer_percent=0.15):
    """Calculates a safe Y position above the max data point."""
    if data_series.empty: return 0
    y_max = data_series.max()
    return y_max + (y_max * buffer_percent)


# ==================================================================================================
# E:I SIGNAL QUALITY FILTERING
# ==================================================================================================

# Minimum unitary (ISI 300) Gabazine amplitude for reliable analysis (mV)
MIN_UNITARY_GABAZINE_AMPLITUDE = 0.5

# Date cutoff for E/I analysis - cells from this date onwards are EXCLUDED
# (20250909 onwards = Stratum Oriens pathway, not Schaffer/Perforant)
# Removed hardcoded date cutoff - now using explicit Pathway column filtering
# EI_DATE_CUTOFF = 20250909 

def _extract_date_from_cell_id(cell_id):
    """Helper to extract date part from cell id like '20240125_c5'"""
    try:
        if isinstance(cell_id, str):
            date_part = cell_id.split('_')[0]
            if date_part.isdigit() and len(date_part) == 8:
                return int(date_part)
    except:
        pass
    return None

def get_valid_cells_for_ei_analysis(df_amplitudes, channel, min_amplitude=MIN_UNITARY_GABAZINE_AMPLITUDE):
    """
    Return list of Cell_IDs for E:I analysis.
    
    MODIFIED: Now includes ALL cells with ANY data (Control or Gabazine)
    MODIFIED: Basal/Stratum Oriens filtering is now explicit via Pathway column
    
    Exclusion criteria:
    1. Cells are designated as 'Basal_Stratum_Oriens' in Pathway column are excluded 
       from Perforant/Schaffer analysis.
    
    Parameters:
        df_amplitudes: DataFrame with E:I amplitude data
        channel: 'channel_1' or 'channel_2' to filter
        min_amplitude: Not used (kept for API compatibility)
    
    Returns:
        List of valid Cell_IDs for this channel
    """
    # Get unitary (ISI 300) data for this channel
    unitary_data = df_amplitudes[(df_amplitudes['Channel'] == channel) & 
                                  (df_amplitudes['ISI'] == 300)]
    
    # Include ALL cells with ANY data (Control OR Gabazine)
    valid_data = unitary_data[
        (unitary_data['Control_Amplitude'].notna()) | 
        (unitary_data['Gabazine_Amplitude'].notna())
    ]
    
    # Explicitly EXCLUDE Basal/Stratum Oriens data from standard channel analysis
    #    (Basal data should be requested specifically via pathway='basal')
    if 'Pathway' in valid_data.columns:
        valid_data = valid_data[valid_data['Pathway'] != 'Basal_Stratum_Oriens']
        
    return valid_data['Cell_ID'].unique().tolist()


def filter_df_by_signal_quality(df, channel, df_amplitudes=None, min_amplitude=MIN_UNITARY_GABAZINE_AMPLITUDE):
    """
    Filter a DataFrame to only include cells with sufficient signal quality.
    
    Parameters:
        df: DataFrame to filter (must have 'Cell_ID' column)
        channel: Channel to check signal quality for
        df_amplitudes: DataFrame with amplitude data (if None, returns df unchanged)
        min_amplitude: Minimum unitary Gabazine amplitude threshold
    
    Returns:
        Filtered DataFrame with only valid cells
    """
    if df_amplitudes is None:
        return df
    
    valid_cells = get_valid_cells_for_ei_analysis(df_amplitudes, channel, min_amplitude)
    return df[df['Cell_ID'].isin(valid_cells)]


# ==================================================================================================
# E:I TRACE PLOTTING FUNCTIONS
# ==================================================================================================

def compute_average_trace(df_subset, trace_col):
    """
    Compute average trace from a subset of the dataframe.
    Uses the most common trace length to avoid artifacts from short traces.
    """
    traces = []
    lengths = []
    for _, row in df_subset.iterrows():
        trace = row[trace_col]
        if isinstance(trace, np.ndarray) and len(trace) > 0:
            traces.append(trace)
            lengths.append(len(trace))
    
    if not traces:
        return None
    
    # Use traces with the most common length (mode) to avoid short artifact traces
    from collections import Counter
    length_counts = Counter(lengths)
    target_length = length_counts.most_common(1)[0][0]
    
    # Filter to only include traces of the target length
    filtered_traces = [t for t in traces if len(t) == target_length]
    
    if not filtered_traces:
        return None
    
    return np.mean(filtered_traces, axis=0)


def plot_ei_averages(ax, df_traces, genotype, isi, label, add_legend=False, pathway='both', max_time_ms=180):
    """
    Plot average traces for Control (black), Gabazine (red), and Expected (grey) for specified pathway.
    Returns (x_max, y_min, y_max) for axis normalization.
    
    Args:
        pathway: 'perforant', 'schaffer', 'basal', or 'both' (default)
    
    Note: Excludes cells from 20250909 onwards for perforant/schaffer (different pathway: Stratum Oriens).
    For basal pathway, uses Pathway column = 'Basal_Stratum_Oriens'.
    """
    # Filter for this condition
    # Robustly match genotype even if already renamed in df
    genotype_targets = [genotype, 'GNB1', 'I80T/+']
    condition = df_traces[(df_traces['Genotype'].isin(genotype_targets)) & 
                         (df_traces['ISI'] == isi)].copy()
    
    # EXCLUDE Basal data if not plotting Basal
    # Basal data (Stratum Oriens) is in Channel 1 but has 'Basal_Stratum_Oriens' in Pathway
    if pathway != 'basal':
        if 'Pathway' in condition.columns:
            condition = condition[condition['Pathway'] != 'Basal_Stratum_Oriens']
    
    if len(condition) == 0:
        geno_label = GENOTYPE_DISPLAY.get(genotype, genotype)
        ax.text(0.5, 0.5, f'No data for {geno_label} {isi}ms ISI', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return None, None, None
    
    # Get data for each pathway
    # For basal, use Pathway column directly
    if pathway == 'basal':
        # Basal pathway uses Pathway column
        basal_data = condition[condition['Pathway'] == 'Basal_Stratum_Oriens']
        perforant_data = pd.DataFrame()  # Empty
        schaffer_data = pd.DataFrame()  # Empty
    else:
        # Schaffer = channel_2, Perforant = channel_1 (apical pathways)
        perforant_data = condition[condition['Channel'] == 'channel_1']
        schaffer_data = condition[condition['Channel'] == 'channel_2']
        basal_data = pd.DataFrame()  # Empty
    
    # Compute average traces
    schaffer_control = compute_average_trace(schaffer_data, 'Control_Trace')
    schaffer_gabazine = compute_average_trace(schaffer_data, 'Gabazine_Trace')
    schaffer_expected = compute_average_trace(schaffer_data, 'Expected_EPSP_Trace')
    perforant_control = compute_average_trace(perforant_data, 'Control_Trace')
    perforant_gabazine = compute_average_trace(perforant_data, 'Gabazine_Trace')
    perforant_expected = compute_average_trace(perforant_data, 'Expected_EPSP_Trace')
    basal_control = compute_average_trace(basal_data, 'Control_Trace')
    basal_gabazine = compute_average_trace(basal_data, 'Gabazine_Trace')
    basal_expected = compute_average_trace(basal_data, 'Expected_EPSP_Trace')
    
    # Trim expected traces to match control trace length - trim from END to preserve timing
    if schaffer_control is not None and schaffer_expected is not None:
        control_len = len(schaffer_control)
        if len(schaffer_expected) > control_len:
            schaffer_expected = schaffer_expected[:control_len]
    
    if perforant_control is not None and perforant_expected is not None:
        control_len = len(perforant_control)
        if len(perforant_expected) > control_len:
            perforant_expected = perforant_expected[:control_len]
    
    if basal_control is not None and basal_expected is not None:
        control_len = len(basal_control)
        if len(basal_expected) > control_len:
            basal_expected = basal_expected[:control_len]
    
    # Slice all traces 160ms shorter (3200 samples at 20kHz)
    slice_samples = 3200
    for trace_var in ['schaffer_control', 'schaffer_gabazine', 'schaffer_expected',
                      'perforant_control', 'perforant_gabazine', 'perforant_expected',
                      'basal_control', 'basal_gabazine', 'basal_expected']:
        trace = locals()[trace_var]
        if trace is not None and len(trace) > slice_samples:
            locals()[trace_var] = trace[:-slice_samples]
    
    # Extract sliced traces back
    schaffer_control, schaffer_gabazine, schaffer_expected = locals()['schaffer_control'], locals()['schaffer_gabazine'], locals()['schaffer_expected']
    perforant_control, perforant_gabazine, perforant_expected = locals()['perforant_control'], locals()['perforant_gabazine'], locals()['perforant_expected']
    basal_control, basal_gabazine, basal_expected = locals()['basal_control'], locals()['basal_gabazine'], locals()['basal_expected']
    
    # Align all traces to baseline (first 100 samples) so they all start at same level
    baseline_samples = 100

def plot_unitary_breakdown(ax, df_traces, genotype, pathway_label, annotate=True):
    """
    Plot unitary breakdown trace (average ± SEM, 300ms ISI) with component identification.
    
    Args:
        ax: Matplotlib axis
        df_traces: Traces dataframe
        genotype: 'WT' or 'I80T/+' (or 'GNB1')
        pathway_label: e.g. 'ECIII Input', 'CA3 Apical Input', 'CA3 Basal Input'
        annotate: Boolean, whether to add brackets and component labels
    """
    isi = 300
    pathway_map = {
        'ECIII (Perforant)': 'perforant',
        'CA3 Apical (Schaffer)': 'schaffer',
        'CA3 Basal (Basal)': 'basal',
        # Display label aliases
        'ECIII Input': 'perforant',
        'CA3 Apical Input': 'schaffer',
        'CA3 Basal Input': 'basal',
    }
    pathway_key = pathway_map.get(pathway_label, 'perforant')

    # --- Genotype filter: WT is WT only; mutant includes both naming conventions ---
    if genotype in ('GNB1', 'I80T/+'):
        genotype_targets = ['GNB1', 'I80T/+']
    else:
        genotype_targets = ['WT']

    condition = df_traces[
        (df_traces['Genotype'].isin(genotype_targets)) &
        (df_traces['ISI'] == isi)
    ].copy()

    # --- Pathway filter ---
    if pathway_key == 'basal':
        subset = condition[condition['Pathway'] == 'Basal_Stratum_Oriens']
    elif pathway_key == 'perforant':
        subset = condition[condition['Channel'] == 'channel_1']
        if 'Pathway' in subset.columns:
            subset = subset[subset['Pathway'] != 'Basal_Stratum_Oriens']
    else:  # schaffer
        subset = condition[condition['Channel'] == 'channel_2']

    if len(subset) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='red', fontsize=8)
        ax.axis('off')
        return None

    # --- Helper: compute mean + SEM from a trace column ---
    def compute_mean_sem(df_sub, col):
        traces = []
        for _, row in df_sub.iterrows():
            t = row[col]
            if isinstance(t, np.ndarray) and len(t) > 0:
                traces.append(t)
        if not traces:
            return None, None
        from collections import Counter
        lengths = [len(t) for t in traces]
        target_len = Counter(lengths).most_common(1)[0][0]
        traces = np.array([t for t in traces if len(t) == target_len])
        mean = np.mean(traces, axis=0)
        sem  = np.std(traces, axis=0, ddof=1) / np.sqrt(len(traces))
        return mean, sem

    control,  ctrl_sem  = compute_mean_sem(subset, 'Control_Trace')
    gabazine, gab_sem   = compute_mean_sem(subset, 'Gabazine_Trace')

    if control is None or gabazine is None:
        return None

    # N-count (unique cells)
    n_cells = subset['Cell_ID'].dropna().nunique()

    # Baseline alignment
    baseline_samples = 100
    control  = control  - np.mean(control [:baseline_samples])
    gabazine = gabazine - np.mean(gabazine[:baseline_samples])

    time = np.arange(len(control)) * 1000 / 20000  # 20 kHz → ms

    # --- Plot mean traces + SEM shading ---
    # Gabazine / Excitation (magenta)
    ax.fill_between(time, gabazine - gab_sem, gabazine + gab_sem,
                    color='magenta', alpha=0.2, edgecolor='none')
    ax.plot(time, gabazine, color='magenta', linewidth=1.0,
            label=f'Excitation (n={n_cells})')

    # Control / With Inhibition (black)
    ax.fill_between(time, control - ctrl_sem, control + ctrl_sem,
                    color='black', alpha=0.15, edgecolor='none')
    ax.plot(time, control, color='black', linewidth=1.0,
            label=f'Inh(GABAA) (n={n_cells})')

    # --- Optional annotation of components ---
    if annotate:
        gab_peak_idx = np.nanargmax(gabazine)
        gab_peak_val = gabazine[gab_peak_idx]
        ctrl_peak_val = control[gab_peak_idx]

        label_x = time[-1] + 5  # 5 ms to the right of trace end

        # 1. Excitation arrow
        ax.annotate('', xy=(time[gab_peak_idx], 0),
                    xytext=(time[gab_peak_idx], gab_peak_val),
                    arrowprops=dict(arrowstyle='<->', color='magenta', lw=1.0))
        ax.text(label_x, gab_peak_val / 2, 'Excitation',
                color='magenta', fontsize=7, va='center', fontweight='bold')

        # 2. Inh(GABAA) arrow (peak difference)
        ax.annotate('', xy=(time[gab_peak_idx], ctrl_peak_val),
                    xytext=(time[gab_peak_idx], gab_peak_val),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
        ax.text(label_x, (gab_peak_val + ctrl_peak_val) / 2, 'Inh(GABAA)',
                color='black', fontsize=7, va='center', fontweight='bold')

        # 3. Slow IPSP (GABAB) — shaded area below baseline on gabazine trace
        ax.fill_between(time, gabazine, 0, where=(gabazine < 0),
                        color='gray', alpha=0.35, edgecolor='none')
        neg_indices = np.where(gabazine < 0)[0]
        if len(neg_indices) > 0:
            ax.text(label_x, -0.5, 'Slow IPSP Area\n(GABAB)',
                    ha='left', fontsize=7, color='gray', fontweight='bold')

    # Baseline reference line
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(pathway_label, fontsize=10)
    ax.axis('off')

    return time[0] if len(time) > 0 else 0, 0

    def align_to_baseline(trace):
        if trace is not None and len(trace) > baseline_samples:
            baseline = np.mean(trace[:baseline_samples])
            return trace - baseline
        return trace
    
    
    schaffer_control = align_to_baseline(schaffer_control)
    schaffer_gabazine = align_to_baseline(schaffer_gabazine)
    schaffer_expected = align_to_baseline(schaffer_expected)
    perforant_control = align_to_baseline(perforant_control)
    perforant_gabazine = align_to_baseline(perforant_gabazine)
    perforant_expected = align_to_baseline(perforant_expected)
    basal_control = align_to_baseline(basal_control)
    basal_gabazine = align_to_baseline(basal_gabazine)
    basal_expected = align_to_baseline(basal_expected)
    
    # Determine time offset for Perforant Path
    x_offset = 0
    x_max = 0
    
    # Plot Schaffer Path traces (only if pathway is 'schaffer' or 'both')
    if pathway in ['schaffer', 'both']:
        if schaffer_control is not None:
            time_schaffer = np.arange(len(schaffer_control)) / 20  # 20kHz sampling -> ms
            ax.plot(time_schaffer, schaffer_control, 'k-', linewidth=1.0, label='Measured: With Inhibition')
            x_offset = time_schaffer[-1] + 50  # Gap between pathways
            x_max = time_schaffer[-1]
            
        if schaffer_gabazine is not None:
            time_schaffer = np.arange(len(schaffer_gabazine)) / 20
            ax.plot(time_schaffer, schaffer_gabazine, 'r-', linewidth=1.0, label='Measured: No Inhibition')
            if x_offset == 0:
                x_offset = time_schaffer[-1] + 50
            x_max = max(x_max, time_schaffer[-1])
        
        # Expected traces need time offset to align with measured traces (they have less pre-stim baseline)
        expected_time_offset = 0  # No offset needed: create_expected_EPSP aligns EPSP onset to stim_index
        
        if schaffer_expected is not None:
            time_schaffer = (np.arange(len(schaffer_expected)) / 20) + expected_time_offset
            ax.plot(time_schaffer, schaffer_expected, color='grey', alpha=0.6, linewidth=1.0, label='Expected: Linear Summation')
            x_max = max(x_max, time_schaffer[-1])
    
    # Plot Perforant Path traces (only if pathway is 'perforant' or 'both')
    # If plotting 'both', offset on x-axis; if only perforant, start at 0
    if pathway in ['perforant', 'both']:
        perforant_offset = x_offset if pathway == 'both' else 0
        expected_time_offset = 0  # No offset needed
        
        if perforant_control is not None:
            time_perforant = np.arange(len(perforant_control)) / 20 + perforant_offset
            # Only add label if this is the first set of traces (perforant only mode)
            label_ctrl = 'Measured: With Inhibition' if pathway == 'perforant' else None
            ax.plot(time_perforant, perforant_control, 'k-', linewidth=1.0, label=label_ctrl)
            x_max = max(x_max, time_perforant[-1])
            
        if perforant_gabazine is not None:
            time_perforant = np.arange(len(perforant_gabazine)) / 20 + perforant_offset
            label_gaba = 'Measured: No Inhibition' if pathway == 'perforant' else None
            ax.plot(time_perforant, perforant_gabazine, 'r-', linewidth=1.0, label=label_gaba)
            x_max = max(x_max, time_perforant[-1])
        
        if perforant_expected is not None:
            time_perforant = (np.arange(len(perforant_expected)) / 20) + perforant_offset + expected_time_offset
            label_exp = 'Expected: Linear Summation' if pathway == 'perforant' else None
            ax.plot(time_perforant, perforant_expected, color='grey', alpha=0.6, linewidth=1.0, label=label_exp)
        x_max = max(x_max, time_perforant[-1])
    
    
    # Plot Basal (Stratum Oriens) pathway traces
    if pathway == 'basal':
        expected_time_offset = 0  # No offset needed
        
        if basal_control is not None:
            time_basal = np.arange(len(basal_control)) / 20
            ax.plot(time_basal, basal_control, 'k-', linewidth=1.0, label='Control')
            x_max = time_basal[-1]
        
        if basal_gabazine is not None:
            time_basal = np.arange(len(basal_gabazine)) / 20
            ax.plot(time_basal, basal_gabazine, 'r-', linewidth=1.0, label='Gabazine')
            x_max = max(x_max, time_basal[-1])
        
        if basal_expected is not None:
            time_basal = (np.arange(len(basal_expected)) / 20) + expected_time_offset
            ax.plot(time_basal, basal_expected, color='grey', alpha=0.6, linewidth=1.0, label='Expected')
            x_max = max(x_max, time_basal[-1])
    # Style the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add labels
    ax.text(0.05, 0.95, label, transform=ax.transAxes, 
           verticalalignment='top', fontweight='bold')
    
    
    # Add legend if requested
    if add_legend:
        ax.legend(frameon=False, loc='upper right')
    
    # Limit x-axis to zoom in on the relevant part of the trace (EPSP)
    #ax.set_xlim(0, max_time_ms)
    
    # Get current y limits
    y_min, y_max = ax.get_ylim()
    
    # Add scale bar
    # 50ms, 2mV hardcoded for consistency
    # Position adjusted to fit within the shortened window
    add_scale_bar(ax, x_scale_ms=50, y_scale_mv=2, x_pos=0.65, y_pos=0.1)
    
    return max_time_ms, y_min, y_max


def add_scale_bar(ax, x_scale_ms, y_scale_mv, x_pos=0.85, y_pos=0.15):
    """
    Add a scale bar to an axes.
    
    Parameters:
        ax: matplotlib axes
        x_scale_ms: horizontal scale bar length in ms
        y_scale_mv: vertical scale bar length in mV
        x_pos, y_pos: position in axes coordinates (0-1)
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate position in data coordinates
    x_start = xlim[0] + x_pos * (xlim[1] - xlim[0])
    y_start = ylim[0] + y_pos * (ylim[1] - ylim[0])
    
    # Draw horizontal bar (time)
    ax.plot([x_start, x_start + x_scale_ms], [y_start, y_start], 
           'k-', linewidth=1, clip_on=False)
    
    # Draw vertical bar (voltage)
    ax.plot([x_start, x_start], [y_start, y_start + y_scale_mv], 
           'k-', linewidth=1, clip_on=False)
    
    # Add labels
    ax.text(x_start + x_scale_ms/2, y_start - 0.5, f'{int(x_scale_ms)} ms', 
           ha='center', va='top')
    ax.text(x_start - 5, y_start + y_scale_mv/2, f'{int(y_scale_mv)} mV', 
           ha='right', va='center', rotation=90)


def plot_epsp_amplitudes(ax, df_amplitudes, channel, genotype, title=None):
    """
    Plot EPSP amplitudes (Control, Gabazine, Expected) vs ISI for a given channel and genotype.
    
    Parameters:
        ax: matplotlib axes
        df_amplitudes: DataFrame with amplitude data
        channel: 'channel_1' (Perforant) or 'channel_2' (Schaffer)
        genotype: 'WT' or 'GNB1'
        title: Optional title for the plot
    
    Note: Automatically excludes cells with weak unitary Gabazine signal (< 0.5 mV)
    """
    # Load statistical results
    stats_gabazine_vs_control = None
    stats_gabazine_vs_expected = None
    
    # Map channel to pathway name for stats lookup
    pathway_map = {'channel_1': 'Perforant', 'channel_2': 'Schaffer'}
    pathway = pathway_map.get(channel, channel)
    
    # Load Drug effect stats (Control vs Gabazine within genotype)
    stats_file_1 = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Drug_PostHoc.csv'
    if os.path.exists(stats_file_1):
        df_stats_1 = pd.read_csv(stats_file_1)
        stats_gabazine_vs_control = df_stats_1[
            (df_stats_1['Pathway'] == pathway) & 
            (df_stats_1['Genotype'] == genotype)
        ]
    
    # Load Gabazine vs Expected stats
    stats_file_2 = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_GabazineVsExpected_PostHoc.csv'
    if os.path.exists(stats_file_2):
        df_stats_2 = pd.read_csv(stats_file_2)
        stats_gabazine_vs_expected = df_stats_2[
            (df_stats_2['Pathway'] == pathway) & 
            (df_stats_2['Genotype'] == genotype)
        ]
    
    # Apply signal quality filter
    valid_cells = get_valid_cells_for_ei_analysis(df_amplitudes, channel)
    
    # Filter data
    subset = df_amplitudes[(df_amplitudes['Channel'] == channel) & 
                           (df_amplitudes['Genotype'] == genotype) &
                           (df_amplitudes['Cell_ID'].isin(valid_cells))]
    
    if subset.empty:
        ax.text(0.5, 0.5, f'No data for {channel} {genotype}', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # ISI order for x-axis (reversed for decreasing order like the traces)
    isi_order = [300, 100, 50, 25, 10]
    x_positions = range(len(isi_order))
    
    # Calculate mean and SEM for each condition at each ISI
    def calc_stats(data, col):
        means = []
        sems = []
        for isi in isi_order:
            isi_data = data[data['ISI'] == isi][col].dropna()
            if len(isi_data) > 0:
                means.append(isi_data.mean())
                sems.append(isi_data.std(ddof=1) / np.sqrt(len(isi_data)) if len(isi_data) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        return means, sems
    
    # Calculate stats for each condition
    control_means, control_sems = calc_stats(subset, 'Control_Amplitude')
    gabazine_means, gabazine_sems = calc_stats(subset, 'Gabazine_Amplitude')
    expected_means, expected_sems = calc_stats(subset, 'Expected_EPSP_Amplitude')
    
    # Plot with error bars
    ax.errorbar(x_positions, control_means, yerr=control_sems, 
               fmt='o-', color='black', linewidth=1, markersize=3, 
               capsize=2, label='Control')
    ax.errorbar(x_positions, gabazine_means, yerr=gabazine_sems, 
               fmt='o-', color='red', linewidth=1, markersize=3, 
               capsize=2, label='Gabazine')
    ax.errorbar(x_positions, expected_means, yerr=expected_sems, 
               fmt='o-', color='grey', linewidth=1, markersize=3, 
               capsize=2, label='Expected EPSP')
    
    # Add statistical significance markers
    y_max = ax.get_ylim()[1]
    
    # Helper function to get significance symbol
    def get_sig_symbol(p_value):
        if pd.isna(p_value):
            return ''
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Helper function to get asterisk symbol for Expected comparison (was hash)
    def get_star_symbol_grey(p_value):
        if pd.isna(p_value):
            return ''
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Add asterisks for Gabazine vs Control (above Gabazine line)
    if stats_gabazine_vs_control is not None and not stats_gabazine_vs_control.empty:
        for i, isi in enumerate(isi_order):
            isi_str = f'ISI{isi}'
            matching_rows = stats_gabazine_vs_control[stats_gabazine_vs_control['ISI_Time'] == isi_str]
            if not matching_rows.empty:
                p_val = matching_rows.iloc[0]['p.value']
                symbol = get_sig_symbol(p_val)
                if symbol:
                    # Place above gabazine point
                    y_pos = gabazine_means[i] + gabazine_sems[i] + y_max * 0.03
                    ax.text(i, y_pos, symbol, ha='center', va='bottom', 
                           fontweight='bold', color='red')
    
    # Add hash symbols for Gabazine vs Expected (above Expected line)
    if stats_gabazine_vs_expected is not None and not stats_gabazine_vs_expected.empty:
        for i, isi in enumerate(isi_order):
            isi_str = f'ISI{isi}'
            matching_rows = stats_gabazine_vs_expected[stats_gabazine_vs_expected['ISI_Time'] == isi_str]
            if not matching_rows.empty:
                p_val = matching_rows.iloc[0]['p.value']
                symbol = get_star_symbol_grey(p_val)
                if symbol:
                    # Place above expected point
                    y_pos = expected_means[i] + expected_sems[i] + y_max * 0.03
                    ax.text(i, y_pos, symbol, ha='center', va='bottom', 
                           fontweight='bold', color='grey')
    
    # Style the axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(isi) for isi in isi_order])
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('EPSP\nAmplitude (mV)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if title:
        ax.set_title(title, fontweight='bold')


def plot_supralinearity(ax, df_amplitudes, channel, supralin_type='Gabazine', title=None):
    """
    Plot supralinearity (Measured - Expected EPSP) vs ISI for both WT and GNB1.
    
    Parameters:
        ax: matplotlib axes
        df_amplitudes: DataFrame with amplitude data
        channel: 'channel_1' (Perforant) or 'channel_2' (Schaffer)
        supralin_type: 'Control' or 'Gabazine' (default: 'Gabazine')
        title: Optional title for the plot
    
    Note: Automatically excludes cells with weak unitary Gabazine signal (< 0.5 mV)
    """
    # Map channel to pathway name for stats
    pathway_map = {'channel_1': 'Perforant', 'channel_2': 'Schaffer'}
    pathway = pathway_map.get(channel, channel)
    
    # Determine column name and stats file based on type
    if supralin_type == 'Control':
        column_name = 'Control_Supralinearity'
        stats_file = 'paper_data/E_I_data/Control_Supralinearity_Stats_Results.csv'
        label_suffix = ' (With Inhibition)'
    else:  # Gabazine
        column_name = 'Gabazine_Supralinearity'
        stats_file = 'paper_data/E_I_data/Gabazine_Supralinearity_Stats_Results.csv'
        label_suffix = ' (No Inhibition)'
    
    # Load stats if available
    stats_data = None
    if os.path.exists(stats_file):
        df_stats = pd.read_csv(stats_file)
        stats_data = df_stats[df_stats['Pathway'] == pathway]
    
    # Apply signal quality filter
    valid_cells = get_valid_cells_for_ei_analysis(df_amplitudes, channel)
    
    # ISI order for x-axis
    isi_order = [300, 100, 50, 25, 10]
    x_positions = range(len(isi_order))
    
    # Plot both genotypes
    colors = {'WT': 'black', 'GNB1': 'red', 'I80T/+': 'red'}
    markers = {'WT': 'o', 'GNB1': 'o', 'I80T/+': 'o'}
    
    for genotype in ['WT', 'GNB1', 'I80T/+']:
        # Filter data
        subset = df_amplitudes[(df_amplitudes['Channel'] == channel) & 
                               (df_amplitudes['Genotype'] == genotype) &
                               (df_amplitudes['Cell_ID'].isin(valid_cells))]
        
        if subset.empty:
            continue
        
        # Calculate supralinearity for each ISI
        means = []
        sems = []
        for isi in isi_order:
            isi_data = subset[subset['ISI'] == isi][column_name].dropna()
            
            if len(isi_data) > 0:
                means.append(isi_data.mean())
                sems.append(isi_data.std(ddof=1) / np.sqrt(len(isi_data)) if len(isi_data) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        # Plot with error bars
        ax.errorbar(x_positions, means, yerr=sems, 
                   fmt=markers[genotype] + '-', color=colors[genotype],
                   linewidth=1, markersize=3, capsize=2, label=genotype)
    
    # Add zero line for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Helper function for significance symbols
    def get_sig_symbol(p_value):
        if pd.isna(p_value):
            return ''
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Add statistical annotations
    if stats_data is not None and not stats_data.empty:
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        
        for i, isi in enumerate(isi_order):
            isi_str = f'ISI{isi}'
            matching_rows = stats_data[stats_data['ISI_Time'] == isi_str]
            if not matching_rows.empty:
                p_val = matching_rows.iloc[0]['p.value']
                symbol = get_sig_symbol(p_val)
                if symbol:
                    y_pos = y_lim[1] + y_range * 0.05
                    ax.text(i, y_pos, symbol, ha='center', va='bottom',
                           fontweight='bold')
    
    # Style the axes
    # Style the axes
    ax.set_xticks(x_positions)
    # Style the axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(isi) for isi in isi_order])
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel(f'Supralinearity\n(mV){label_suffix}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Legend removed for cleaner appearance
    
    if title:
        ax.set_title(title, fontweight='bold')


def plot_ei_imbalance(ax, df_amplitudes, channel, title=None):
    """
    Plot E:I imbalance vs ISI for both WT and GNB1.
    
    Parameters:
        ax: matplotlib axes
        df_amplitudes: DataFrame with amplitude data
        channel: 'channel_1' (Perforant) or 'channel_2' (Schaffer)
        title: Optional title for the plot
    
    Note: Automatically excludes cells with weak unitary Gabazine signal (< 0.5 mV)
    """
    # Map channel to pathway name for stats
    pathway_map = {'channel_1': 'Perforant', 'channel_2': 'Schaffer'}
    pathway = pathway_map.get(channel, channel)
    
    # Load stats if available
    stats_data = None
    stats_file = 'paper_data/E_I_data/EI_Imbalance_Stats_Results.csv'
    if os.path.exists(stats_file):
        df_stats = pd.read_csv(stats_file)
        stats_data = df_stats[df_stats['Pathway'] == pathway]
    
    # Apply signal quality filter
    valid_cells = get_valid_cells_for_ei_analysis(df_amplitudes, channel)
    
    # ISI order for x-axis
    isi_order = [300, 100, 50, 25, 10]
    x_positions = range(len(isi_order))
    
    # Plot both genotypes
    colors = {'WT': 'black', 'GNB1': 'red', 'I80T/+': 'red'}
    markers = {'WT': 'o', 'GNB1': 'o', 'I80T/+': 'o'}
    
    for genotype in ['WT', 'GNB1', 'I80T/+']:
        # Filter data
        subset = df_amplitudes[(df_amplitudes['Channel'] == channel) & 
                               (df_amplitudes['Genotype'] == genotype) &
                               (df_amplitudes['Cell_ID'].isin(valid_cells))]
        
        if subset.empty:
            continue
        
        # Calculate mean and SEM for E:I imbalance at each ISI
        means = []
        sems = []
        for isi in isi_order:
            isi_data = subset[subset['ISI'] == isi]['E_I_Imbalance'].dropna()
            
            if len(isi_data) > 0:
                means.append(isi_data.mean())
                sems.append(isi_data.std(ddof=1) / np.sqrt(len(isi_data)) if len(isi_data) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        # Plot with error bars
        ax.errorbar(x_positions, means, yerr=sems, 
                   fmt=markers[genotype] + '-', color=colors[genotype],
                   linewidth=1, markersize=3, capsize=2, label=genotype)
    
    # Helper function for significance symbols
    def get_sig_symbol(p_value):
        if pd.isna(p_value):
            return ''
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Add statistical annotations
    if stats_data is not None and not stats_data.empty:
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        
        for i, isi in enumerate(isi_order):
            isi_str = f'ISI{isi}'
            matching_rows = stats_data[stats_data['ISI_Time'] == isi_str]
            if not matching_rows.empty:
                p_val = matching_rows.iloc[0]['p.value']
                symbol = get_sig_symbol(p_val)
                if symbol:
                    y_pos = y_lim[1] + y_range * 0.05
                    ax.text(i, y_pos, symbol, ha='center', va='bottom',
                           fontweight='bold')
    
    # Style the axes
    # Style the axes
    ax.set_xticks(x_positions)
    # Style the axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(isi) for isi in isi_order])
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('E:I Imbalance Ratio')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='best')
    
    if title:
        ax.set_title(title, fontweight='bold')


# ==================================================================================================
# PHYSIOLOGY SUMMARY TABLE
# ==================================================================================================

def create_physiology_summary_table(ax, df_intrinsic, df_ap_ahp, df_stats):
    """
    Creates a formatted summary table of physiological properties for Figure 2.
    
    Parameters:
        ax: matplotlib axis
        df_intrinsic: DataFrame with intrinsic properties (from intrinsic_properties.csv)
        df_ap_ahp: DataFrame with AP/AHP properties (from combined_AP_AHP_rheobase_analysis.csv)
        df_stats: DataFrame with statistical results (from Stats_Results_Figure_2.csv)
    """
    ax.axis('off')
    
    # Define property mappings (CSV column -> Display name -> Units)
    property_specs = [
        # Section 1: Physiological Properties
        {'section': None, 'df': df_intrinsic, 'column': 'Vm rest/start (mV)', 'display': 'Resting membrane potential (mV)', 'stats_key': 'Vm Rest'},
        {'section': None, 'df': df_intrinsic, 'column': 'Input_Resistance_MOhm', 'display': 'Input resistance (MΩ)', 'stats_key': 'Input Resistance'},
        {'section': None, 'df': df_intrinsic, 'column': 'Access Resistance (From Whole Cell V-Clamp)', 'display': 'Access resistance (MΩ)', 'stats_key': 'Access Resistance'},
        {'section': None, 'df': df_intrinsic, 'column': 'Voltage_sag', 'display': 'Voltage Sag (%)', 'stats_key': 'Voltage Sag'},
        # Section 2: AP Properties
        {'section': 'Action potential (AP) properties', 'df': df_ap_ahp, 'column': 'Rheobase_Current', 'display': 'Rheobase (pA)', 'stats_key': 'Rheobase'},
        {'section': None, 'df': df_ap_ahp, 'column': 'AP_threshold', 'display': 'Voltage threshold (mV)', 'stats_key': 'AP Threshold'},
        {'section': None, 'df': df_ap_ahp, 'column': 'AP_size', 'display': 'Amplitude (mV)', 'stats_key': 'AP Size'},
        {'section': None, 'df': df_ap_ahp, 'column': 'AP_halfwidth', 'display': 'Halfwidth (ms)', 'stats_key': 'AP Halfwidth'},
        # Section 3: AHP Properties
        {'section': 'After hyperpolarization (AHP) properties', 'df': df_ap_ahp, 'column': 'AHP_size', 'display': 'Amplitude (mV)', 'stats_key': 'AHP Amplitude'},
        {'section': None, 'df': df_ap_ahp, 'column': 'decay_area', 'display': 'Decay area (ms)', 'stats_key': 'AHP Decay'},
    ]
    
    # Build table data
    table_data = []
    
    for spec in property_specs:
        # Section header row
        if spec['section']:
            table_data.append([spec['section'], '', '', '', '', ''])
        
        # Calculate statistics
        df = spec['df']
        col = spec['column']
        
        if df is None or col not in df.columns:
            row = [spec['display'], 'N/A', 'N/A', '-', '-', '-']
        else:
            # WT statistics
            wt_data = df[df['Genotype'] == 'WT'][col].dropna()
            wt_mean = wt_data.mean() if len(wt_data) > 0 else np.nan
            wt_sem = wt_data.sem() if len(wt_data) > 1 else np.nan
            wt_n = len(wt_data)
            
            # GNB1 statistics  
            gnb1_data = df[df['Genotype'].isin(['GNB1', 'I80T/+'])][col].dropna()
            gnb1_mean = gnb1_data.mean() if len(gnb1_data) > 0 else np.nan
            gnb1_sem = gnb1_data.sem() if len(gnb1_data) > 1 else np.nan
            gnb1_n = len(gnb1_data)
            
            # P-value lookup
            p_value = 'N/A'
            if df_stats is not None and spec['stats_key']:
                stat_match = df_stats[df_stats['Comparison'].str.contains(spec['stats_key'], regex=False, na=False)]
                if not stat_match.empty:
                    p_val_raw = stat_match.iloc[0]['P_Value']
                    p_value = f'{p_val_raw:.4f}'
            
            # Format values
            wt_str = f'{wt_mean:.2f} ± {wt_sem:.2f}' if not np.isnan(wt_mean) else 'N/A'
            gnb1_str = f'{gnb1_mean:.2f} ± {gnb1_sem:.2f}' if not np.isnan(gnb1_mean) else 'N/A'
            
            row = [spec['display'], wt_str, gnb1_str, str(wt_n), str(gnb1_n), p_value]
        
        table_data.append(row)
    
    # Create table
    col_labels = ['Physiological Property', 'WT\nmean', 'GNB1\nI80T +/-\nmean', 'WT\nN', 'GNB1\nI80T +/-\nN', 'p-value']
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.1, 0.1, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        elif col == 0 and row > 0 and table_data[row-1][0] in ['Action potential (AP) properties',
                                                                  'After hyperpolarization (AHP) properties']:
            # Section header rows
            cell.set_facecolor('#FFC000')
            cell.set_text_props(weight='bold')
        else:
            # Data rows - light yellow background
            cell.set_facecolor('#FFF2CC')
        
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    
    # Removed title to avoid text appearing in the middle of the figure
    # ax.set_title('Physiological Properties Summary', fontsize=14, fontweight='bold', pad=20)


def export_physiology_summary_table(df_intrinsic, df_ap_ahp, df_stats, output_path):
    """Export physiology summary table to CSV."""
    import pandas as pd
    import numpy as np
    
    # Define property mappings
    property_specs = [
        {'df': df_intrinsic, 'column': 'Vm rest/start (mV)', 'display': 'Resting membrane potential (mV)', 'stats_key': 'Vm Rest'},
        {'df': df_intrinsic, 'column': 'Input_Resistance_MOhm', 'display': 'Input resistance (MΩ)', 'stats_key': 'Input Resistance'},
        {'df': df_intrinsic, 'column': 'Access Resistance (From Whole Cell V-Clamp)', 'display': 'Access resistance (MΩ)', 'stats_key': 'Access Resistance'},
        {'df': df_intrinsic, 'column': 'Voltage_sag', 'display': 'Voltage Sag (%)', 'stats_key': 'Voltage Sag'},
        {'df': df_ap_ahp, 'column': 'Rheobase_Current', 'display': 'Rheobase (pA)', 'stats_key': 'Rheobase'},
        {'df': df_ap_ahp, 'column': 'AP_threshold', 'display': 'Voltage threshold (mV)', 'stats_key': 'AP Threshold'},
        {'df': df_ap_ahp, 'column': 'AP_size', 'display': 'Amplitude (mV)', 'stats_key': 'AP Size'},
        {'df': df_ap_ahp, 'column': 'AP_halfwidth', 'display': 'Halfwidth (ms)', 'stats_key': 'AP Halfwidth'},
        {'df': df_ap_ahp, 'column': 'AHP_size', 'display': 'AHP Amplitude (mV)', 'stats_key': 'AHP Amplitude'},
        {'df': df_ap_ahp, 'column': 'decay_area', 'display': 'AHP Decay area (ms)', 'stats_key': 'AHP Decay'},
    ]
    
    rows = []
    for spec in property_specs:
        df = spec['df']
        column = spec['column']
        display = spec['display']
        stats_key = spec['stats_key']
        
        if df is None or column not in df.columns:
            continue
        
        # Get WT and GNB1 data
        wt_data = df[df['Genotype'] == 'WT'][column].dropna()
        gnb1_data = df[df['Genotype'].isin(['GNB1', 'I80T/+'])][column].dropna()
        
        # Calculate stats
        wt_mean = wt_data.mean() if len(wt_data) > 0 else np.nan
        wt_sem = wt_data.std() / np.sqrt(len(wt_data)) if len(wt_data) > 0 else np.nan
        gnb1_mean = gnb1_data.mean() if len(gnb1_data) > 0 else np.nan
        gnb1_sem = gnb1_data.std() / np.sqrt(len(gnb1_data)) if len(gnb1_data) > 0 else np.nan
        
        # Get p-value
        p_val = np.nan
        if df_stats is not None:
            match = df_stats[df_stats['Comparison'].str.contains(stats_key, regex=False, na=False)]
            if not match.empty:
                p_val = match.iloc[0]['P_Value']
        
        rows.append({
            'Property': display,
            'WT Mean': wt_mean,
            'WT SEM': wt_sem,
            'WT N': len(wt_data),
            'GNB1 Mean': gnb1_mean,
            'GNB1 SEM': gnb1_sem,
            'GNB1 N': len(gnb1_data),
            'p-value': p_val
        })
    
    # Create DataFrame and export
    export_df = pd.DataFrame(rows)
    export_df.to_csv(output_path, index=False)
    

def plot_bar_scatter_fig2(ax, df, column, ylabel, df_stats, stats_key):
    """Plot bar/scatter plot for a single property."""
    import numpy as np
    
    colors = {'WT': 'black', 'GNB1': 'red', 'I80T/+': 'red'}
    genotypes = ['WT', 'I80T/+']
    plot_data = df.dropna(subset=[column])
    
    for i, genotype in enumerate(genotypes):
        data = plot_data[plot_data['Genotype'] == genotype][column]
        color = colors[genotype]
        n = len(data)
        if n > 0:
            mean_val = data.mean()
            sem_val = data.std() / np.sqrt(n)
            
            ax.bar(i, mean_val, 0.7, color=color, alpha=0.6, edgecolor='black', linewidth=1)
            ax.errorbar(i, mean_val, yerr=sem_val, fmt='none', color=color, capsize=1)
            ax.scatter([i]*n, data.values, color=color, s=2, zorder=3)
    
    ax.set_xticks([0, 1])
    wt_n = len(plot_data[plot_data['Genotype'] == 'WT'])
    gnb1_n = len(plot_data[plot_data['Genotype'].isin(['GNB1', 'I80T/+'])])
    ax.set_xticklabels([f'WT\n(N={wt_n})', f'GNB1\n(N={gnb1_n})'])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    apply_clean_yticks(ax)
    
    # Add stats annotation
    if df_stats is not None:
        match = df_stats[df_stats['Comparison'].str.contains(stats_key, regex=False, na=False)]
        if not match.empty:
            p_val = match.iloc[0]['P_Value']
            if p_val < 0.05:
                y_max = plot_data[column].max()
                y_pos = y_max * 1.15
                ax.plot([0, 0, 1, 1], [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 'k-', linewidth=1)
                
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                else:
                    sig = '*'
                ax.text(0.5, y_pos*1.03, sig, ha='center', va='bottom',
                       fontsize=14, fontweight='bold')


def plot_isi_example_traces(ax_wt, ax_gnb1, data_dir, master_df, df_ap_ahp, target_wt='03142024_c2', target_gnb1='02132024_c1'):
    """Plot example traces with 6 spikes showing ISI intervals from specific cells."""
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    
    def find_6spike_trace_from_cell(cell_id, data_dir):
        """Find a 6-spike trace from a specific cell."""
        file_path = find_file_for_cell(data_dir, cell_id)
        if not file_path:
            print(f"File not found for cell {cell_id}")
            return None, None, None
        
        try:
            data_df = pd.read_pickle(file_path)
            # Find Coarse_FI or IV_stim traces
            fi_data = data_df[data_df['stim_type'].isin(['Coarse_FI', 'IV_stim'])]
            
            for idx, row in fi_data.iterrows():
                trace = row['sweep']
                # Find peaks
                peaks, _ = find_peaks(trace, height=-20, distance=100)
                
                if len(peaks) == 6:  # Found trace with exactly 6 spikes
                    # Check if last spike is not too close to end (allow 100ms buffer)
                    sampling_rate = 20000
                    dt_ms = 1000 / sampling_rate
                    peak_times = peaks * dt_ms
                    last_peak_time = peak_times[-1]
                    trace_end_time = len(trace) * dt_ms
                    
                    if trace_end_time - last_peak_time > 100:  # At least 100ms after last spike
                        return trace, peaks, cell_id
        except Exception as e:
            print(f"Error loading trace for {cell_id}: {e}")
            return None, None, None
        
        return None, None, None
    
    # Find traces for specific cells
    wt_trace, wt_peaks, wt_cell = find_6spike_trace_from_cell(target_wt, data_dir)
    gnb1_trace, gnb1_peaks, gnb1_cell = find_6spike_trace_from_cell(target_gnb1, data_dir)
    
    sampling_rate = 20000
    dt_ms = 1000 / sampling_rate
    
    # Plot WT - with baseline normalization
    if wt_trace is not None:
        time = np.arange(len(wt_trace)) * dt_ms
        
        # Normalize to common baseline
        baseline_wt = np.mean(wt_trace[int(0.1*len(wt_trace)):int(0.15*len(wt_trace))])
        common_baseline = -65  # mV
        wt_trace_aligned = wt_trace - baseline_wt + common_baseline
        
        ax_wt.plot(time, wt_trace_aligned, 'k-', linewidth=1)
        
        # Mark peaks
        peak_times = wt_peaks * dt_ms
        peak_voltages = wt_trace_aligned[wt_peaks]
        # REMOVED: Peak scatter points for cleaner visualization
        # ax_wt.scatter(peak_times, peak_voltages, color='red', s=2, zorder=5)
        
        # REMOVED: ISI interval lines and labels for cleaner visualization
        # for i in range(len(wt_peaks)-1):
        #     t1, t2 = peak_times[i], peak_times[i+1]
        #     v_line = peak_voltages[i:i+2].max() + 5
        #     ax_wt.plot([t1, t2], [v_line, v_line], 'b-', linewidth=1)
        #     # REMOVED: ISI time labels for cleaner visualization
        #     # isi_ms = t2 - t1
        #     # ax_wt.text((t1+t2)/2, v_line+2, f'{isi_ms:.0f}ms',  
        #     #           ha='center', va='bottom', fontsize=7, color='blue')
        
        ax_wt.text(0.02, 0.95, 'WT', transform=ax_wt.transAxes, 
                  fontsize=10, fontweight='bold', va='top')
        ax_wt.set_xlim(150, time[wt_peaks[-1]] + 100)
    else:
        ax_wt.text(0.5, 0.5, f'WT: No 6-spike trace found for {target_wt}', ha='center')
    
    ax_wt.axis('off')
    
    # Plot GNB1 - with baseline normalization
    if gnb1_trace is not None:
        time = np.arange(len(gnb1_trace)) * dt_ms
        
        # Normalize to same common baseline
        baseline_gnb1 = np.mean(gnb1_trace[int(0.1*len(gnb1_trace)):int(0.15*len(gnb1_trace))])
        common_baseline = -65  # mV
        gnb1_trace_aligned = gnb1_trace - baseline_gnb1 + common_baseline
        
        ax_gnb1.plot(time, gnb1_trace_aligned, 'r-', linewidth=1)
        
        # Mark peaks
        peak_times = gnb1_peaks * dt_ms
        peak_voltages = gnb1_trace_aligned[gnb1_peaks]
        # REMOVED: Peak scatter points for cleaner visualization
        # ax_gnb1.scatter(peak_times, peak_voltages, color='red', s=2, zorder=5)
        
        # REMOVED: ISI interval lines and labels for cleaner visualization
        # for i in range(len(gnb1_peaks)-1):
        #     t1, t2 = peak_times[i], peak_times[i+1]
        #     v_line = peak_voltages[i:i+2].max() + 5
        #     ax_gnb1.plot([t1, t2], [v_line, v_line], 'b-', linewidth=1)
        #     # REMOVED: ISI time labels for cleaner visualization
        #     # isi_ms = t2 - t1
        #     # ax_gnb1.text((t1+t2)/2, v_line+2, f'{isi_ms:.0f}ms',
        #     #             ha='center', va='bottom', fontsize=7, color='blue')
        
        ax_gnb1.text(0.02, 0.95, 'I80T/+', transform=ax_gnb1.transAxes,
                    fontsize=10, fontweight='bold', va='top', color='red')
        ax_gnb1.set_xlim(150, time[gnb1_peaks[-1]] + 100)
    else:
        ax_gnb1.text(0.5, 0.5, f'GNB1: No 6-spike trace found for {target_gnb1}', ha='center')
    
    ax_gnb1.axis('off')
    
    # Add scale bars to GNB1 plot (100ms, 20mV)
    if gnb1_trace is not None:
        add_scale_bar(ax_gnb1, 100, 20, x_pos=0.85, y_pos=0.1)


# ==================================================================================================
# FIGURE 2 PANEL UPDATES - Voltage Sag and AHP Area Comparisons
# ==================================================================================================

def plot_voltage_sag_comparison(ax_wt, ax_gnb1, data_dir, master_df, target_wt='03142024_c2', target_gnb1='02132024_c1'):
    """Plot WT and GNB1 voltage sag examples side-by-side with analysis annotations on WT."""
    import pandas as pd
    import numpy as np
    
    def plot_single_voltage_sag(ax, cell_id, genotype, add_annotations=False):
        file_path = find_file_for_cell(data_dir, cell_id)
        if not file_path:
            plot_trace_placeholder(ax, f"File not found: {cell_id}")
            return
        
        try:
            data_df = pd.read_pickle(file_path)
            sag_data = data_df[data_df['stim_type'] == 'Voltage_sag']
            if sag_data.empty:
                plot_trace_placeholder(ax, "No Voltage_sag experiment found")
                return
            
            trace = sag_data.iloc[0]['sweep']
            sampling_rate = 20000
            dt_ms = 1000 / sampling_rate
            time = np.arange(len(trace)) * dt_ms
            
            start_time_ms, end_time_ms = 350, 850
            start_idx = int(start_time_ms * sampling_rate / 1000)
            end_idx = int(end_time_ms * sampling_rate / 1000)
            baseline_start_idx = start_idx - int(50 * sampling_rate / 1000)
            
            baseline_voltage = np.mean(trace[baseline_start_idx:start_idx])
            min_voltage = np.min(trace[start_idx:end_idx])
            steady_state_voltage = np.mean(trace[end_idx - int(50 * sampling_rate / 1000):end_idx])
            
            color = 'black' if genotype == 'WT' else 'red'
            ax.plot(time, trace, color=color, linewidth=1.5)
            
            if add_annotations:
                ax.axhline(y=baseline_voltage, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.axhline(y=min_voltage, color='blue', linestyle=':', linewidth=0.8, alpha=0.8)
                ax.axhline(y=steady_state_voltage, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.text(250, baseline_voltage+1.5, '$V_{baseline}$', fontsize=9, color='gray', fontweight='bold')
                ax.text(500, min_voltage-2, '$V_{min}$', fontsize=9, color='blue', fontweight='bold')
                ax.text(end_time_ms-100, steady_state_voltage+1.5, '$V_{steady}$', fontsize=9, color='green', fontweight='bold')
            
            label_color = 'black' if genotype == 'WT' else 'red'
            ax.text(0.02, 0.95, genotype, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', color=label_color)
            ax.set_xlim(200, 950)
            ax.axis('off')
            if genotype in ('GNB1', 'I80T/+'):
                add_scale_bar(ax, 100, 10, x_pos=0.8, y_pos=0.15)
        except Exception as e:
            print(f"Error plotting voltage sag for {cell_id}: {e}")
            plot_trace_placeholder(ax, f"Error: {e}")
    
    plot_single_voltage_sag(ax_wt, target_wt, 'WT', add_annotations=False)
    plot_single_voltage_sag(ax_gnb1, target_gnb1, 'I80T/+', add_annotations=False)


def plot_input_resistance_comparison(ax_wt, ax_gnb1, data_dir, master_df, target_wt='03142024_c2', target_gnb1='02132024_c1'):
    """Plot WT and GNB1 input resistance examples side-by-side."""
    import pandas as pd
    import numpy as np
    
    def plot_single_input_resistance(ax, cell_id, genotype):
        file_path = find_file_for_cell(data_dir, cell_id)
        if not file_path:
            plot_trace_placeholder(ax, f"File not found")
            return
        
        try:
            data_df = pd.read_pickle(file_path)
            
            trace = None
            for stim_type in ['Voltage_sag', 'EPSP_stim', 'IV_stim', 'Coarse_FI', 'Fine_FI']:
                sag_data = data_df[data_df['stim_type'] == stim_type]
                if not sag_data.empty:
                    trace = sag_data.iloc[0]['sweep']
                    break
                    
            if trace is None:
                plot_trace_placeholder(ax, "Missing Trace")
                return
            sampling_rate = 20000
            dt_ms = 1000 / sampling_rate
            time = np.arange(len(trace)) * dt_ms
            
            color = 'black' if genotype == 'WT' else 'red'
            ax.plot(time, trace, color=color, linewidth=1.5)
            
            label_color = 'black' if genotype == 'WT' else 'red'
            ax.text(0.02, 0.95, genotype, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', color=label_color)
            
            # Zoom in on test pulse
            ax.set_xlim(0, 300)
            ax.axis('off')
            
            if genotype in ('GNB1', 'I80T/+'):
                add_scale_bar(ax, 50, 5, x_pos=0.8, y_pos=0.15)
        except Exception as e:
            plot_trace_placeholder(ax, f"Error: {e}")
    
    plot_single_input_resistance(ax_wt, target_wt, 'WT')
    plot_single_input_resistance(ax_gnb1, target_gnb1, 'I80T/+')



def plot_ahp_area_comparison(ax_wt, ax_gnb1, data_dir, master_df, df_ap_ahp, target_wt='03142024_c2', target_gnb1='02132024_c1'):
    """Plot WT and GNB1 AHP area examples side-by-side showing AP cut off at threshold."""
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    
    def plot_single_ahp_area(ax, cell_id, genotype, df_ap_ahp):
        file_path = find_file_for_cell(data_dir, cell_id)
        if not file_path:
            plot_trace_placeholder(ax, f"File not found: {cell_id}")
            return
        
        try:
            data_df = pd.read_pickle(file_path)
            rheobase_data = data_df[data_df['stim_type'].isin(['Coarse_FI', 'IV_stim'])]
            if rheobase_data.empty:
                plot_trace_placeholder(ax, "No rheobase data")
                return
            
            trace = None
            for idx, row in rheobase_data.iterrows():
                test_trace = row['sweep']
                peaks, _ = find_peaks(test_trace, height=-20, distance=100)
                if len(peaks) >= 1:
                    trace = test_trace
                    break
            
            if trace is None:
                plot_trace_placeholder(ax, "No AP found")
                return
            
            sampling_rate = 20000
            dt_ms = 1000 / sampling_rate
            time = np.arange(len(trace)) * dt_ms
            
            cell_data = df_ap_ahp[df_ap_ahp['Cell_ID'].str.strip() == cell_id.strip()]
            if cell_data.empty:
                plot_trace_placeholder(ax, "No analysis data")
                return
            
            ap_threshold = cell_data.iloc[0]['AP_threshold']
            peaks, _ = find_peaks(trace, height=-20, distance=100)
            peak_idx = peaks[0]
            peak_time = peak_idx * dt_ms
            
            search_window = int(50 / dt_ms)
            trough_idx = peak_idx + np.argmin(trace[peak_idx:peak_idx+search_window])
            trough_voltage = trace[trough_idx]
            trough_time = trough_idx * dt_ms
            
            recovery_idx = trough_idx
            for i in range(trough_idx, min(trough_idx + search_window*2, len(trace))):
                if trace[i] >= ap_threshold:
                    recovery_idx = i
                    break
            
            color = 'black' if genotype == 'WT' else 'red'
            ax.plot(time[:peak_idx], trace[:peak_idx], color=color, linewidth=1.5)
            ax.plot([peak_time - 2, peak_time + 2], [ap_threshold, ap_threshold], color=color, linewidth=2)
            ax.plot(time[peak_idx:recovery_idx], trace[peak_idx:recovery_idx], color=color, linewidth=1.5)
            ax.fill_between(time[peak_idx:recovery_idx], trace[peak_idx:recovery_idx], ap_threshold, color=color, alpha=0.3)
            ax.plot(peak_time, ap_threshold, 'o', color='blue', markersize=5)
            ax.plot(trough_time, trough_voltage, 'o', color='red', markersize=5)
            
            label_color = 'black' if genotype == 'WT' else 'red'
            ax.text(0.02, 0.95, genotype, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', color=label_color)
            ax.set_xlim(peak_time - 10, min(trough_time + 80, time[recovery_idx] + 20))
            ax.axis('off')
            if genotype in ('GNB1', 'I80T/+'):
                add_scale_bar(ax, 20, 20, x_pos=0.75, y_pos=0.15)
        except Exception as e:
            print(f"Error plotting AHP area for {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            plot_trace_placeholder(ax, f"Error: {e}")
    
    plot_single_ahp_area(ax_wt, target_wt, 'WT', df_ap_ahp)
    plot_single_ahp_area(ax_gnb1, target_gnb1, 'I80T/+', df_ap_ahp)

# ==================================================================================================
# FIGURE 4 and 5 HELPER FUNCTIONS FOR 3-PATHWAY LAYOUT
# ==================================================================================================

def plot_epsp_amplitudes_pathway(ax, df_amplitudes, pathway_name, genotype, title=None):
    """
    Plot EPSP amplitudes for a specific pathway (by Pathway column name).
    Used for basal pathway which uses 'Basal_Stratum_Oriens' in Pathway column.
    
    Plots Control (black), Gabazine (red), Expected (grey) lines.
    """
    # Filter data by Pathway column
    subset = df_amplitudes[
        (df_amplitudes['Pathway'] == pathway_name) &
        (df_amplitudes['Genotype'] == genotype)
    ]
    
    if subset.empty:
        ax.text(0.5, 0.5, f'No data for {pathway_name} {genotype}',
               ha='center', va='center', transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # ISI values (reversed for consistency)
    isi_order = [300, 100, 50, 25, 10]
    x_positions = range(len(isi_order))
    
    # Calculate means and SEMs
    def calc_stats(data, col):
        means = []
        sems = []
        for isi in isi_order:
            isi_data = data[data['ISI'] == isi][col].dropna()
            if len(isi_data) > 0:
                means.append(isi_data.mean())
                sems.append(isi_data.std(ddof=1) / np.sqrt(len(isi_data)) if len(isi_data) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        return means, sems
    
    ctrl_means, ctrl_sems = calc_stats(subset, 'Control_Amplitude')
    gab_means, gab_sems = calc_stats(subset, 'Gabazine_Amplitude')
    exp_means, exp_sems = calc_stats(subset, 'Expected_EPSP_Amplitude')
    
    # Plot lines consistent with other pathways
    ax.errorbar(x_positions, ctrl_means, yerr=ctrl_sems,
           fmt='o-', color='black', label='Control', capsize=2, linewidth=1, markersize=3)
    ax.errorbar(x_positions, gab_means, yerr=gab_sems,
           fmt='o-', color='red', label='Gabazine', capsize=2, linewidth=1, markersize=3)
    ax.errorbar(x_positions, exp_means, yerr=exp_sems,
           fmt='o-', color='grey', label='Expected', capsize=2, linewidth=1, markersize=3)
    
    # Format axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(isi) for isi in isi_order])
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if title:
        ax.set_title(title)

    # --- Add Statistics ---
    
    # Map pathway name to what is in the CSV (Basal_Stratum_Oriens -> Stratum_Oriens)
    stats_pathway = 'Stratum_Oriens' if pathway_name == 'Basal_Stratum_Oriens' else pathway_name
    
    # Helper function to get significance symbol
    def get_sig_symbol(p_value):
        if pd.isna(p_value): return ''
        elif p_value < 0.001: return '***'
        elif p_value < 0.01: return '**'
        elif p_value < 0.05: return '*'
        return ''
    
    # Helper for star symbol (Expected comparison) - was hash
    def get_star_symbol_grey(p_value):
        if pd.isna(p_value): return ''
        elif p_value < 0.001: return '***'
        elif p_value < 0.01: return '**'
        elif p_value < 0.05: return '*'
        return ''

    # Load Drug effect stats (Control vs Gabazine within genotype)
    stats_file_1 = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_Drug_PostHoc.csv'
    if os.path.exists(stats_file_1):
        df_stats_1 = pd.read_csv(stats_file_1)
        stats_gab_vs_ctrl = df_stats_1[
            (df_stats_1['Pathway'] == stats_pathway) & 
            (df_stats_1['Genotype'] == genotype)
        ]
        
        # Plot Asterisks
        if not stats_gab_vs_ctrl.empty:
            y_max = ax.get_ylim()[1]
            for i, isi in enumerate(isi_order):
                isi_str = f'ISI{isi}'
                match = stats_gab_vs_ctrl[stats_gab_vs_ctrl['ISI_Time'] == isi_str]
                if not match.empty:
                    p_val = match.iloc[0]['p.value']
                    symbol = get_sig_symbol(p_val)
                    if symbol:
                        # Place above gabazine point
                        y_pos = gab_means[i] + gab_sems[i] + y_max * 0.05
                        ax.text(i, y_pos, symbol, ha='center', va='bottom', 
                               fontweight='bold', color='red')

    # Load Gabazine vs Expected stats
    stats_file_2 = 'paper_data/E_I_data/Figure_4_EI_Amplitudes_GabazineVsExpected_PostHoc.csv'
    if os.path.exists(stats_file_2):
        df_stats_2 = pd.read_csv(stats_file_2)
        stats_gab_vs_exp = df_stats_2[
            (df_stats_2['Pathway'] == stats_pathway) & 
            (df_stats_2['Genotype'] == genotype)
        ]
        
        # Plot Hashes
        if not stats_gab_vs_exp.empty:
            y_max = ax.get_ylim()[1]
            for i, isi in enumerate(isi_order):
                isi_str = f'ISI{isi}'
                match = stats_gab_vs_exp[stats_gab_vs_exp['ISI_Time'] == isi_str]
                if not match.empty:
                    p_val = match.iloc[0]['p.value']
                    symbol = get_star_symbol_grey(p_val)
                    if symbol:
                        # Place above expected point (or gabazine if higher overlap, but keep simple)
                        y_pos = exp_means[i] + exp_sems[i] + y_max * 0.05
                        ax.text(i, y_pos, symbol, ha='center', va='bottom', 
                               fontweight='bold', color='grey')

def plot_gabazine_genotype_comparison(ax, df_amplitudes, pathway_name):
    """
    Plot Gabazine condition only, comparing WT vs GNB1.
    
    Uses significance markers from Figure_5_Significance_Markers.csv:
    - '*' over ISIs with significant post-hoc (FDR corrected)
    - '#' with brackets for significant Genotype:ISI_Time interaction
    
    Args:
        ax: matplotlib axes
        df_amplitudes: DataFrame with E:I amplitudes
        pathway_name: 'Perforant', 'Schaffer', or 'Basal_Stratum_Oriens'
        df_stats: (deprecated, kept for compatibility) - now uses markers file
    """
    # ISI values
    isis = [300, 100, 50, 25, 10]
    
    # Collect data for WT and GNB1 using Gabazine_Amplitude column directly
    wt_means = []
    wt_sems = []
    gnb1_means = []
    gnb1_sems = []
    
    for isi in isis:
        # WT data
        wt_data = df_amplitudes[
            (df_amplitudes['Genotype'] == 'WT') &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ]['Gabazine_Amplitude'].dropna()
        wt_means.append(wt_data.mean() if len(wt_data) > 0 else np.nan)
        wt_sems.append(wt_data.sem() if len(wt_data) > 0 else 0)
        
        # GNB1 data
        gnb1_data = df_amplitudes[
            (df_amplitudes['Genotype'].isin(['GNB1', 'I80T/+'])) &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ]['Gabazine_Amplitude'].dropna()
        gnb1_means.append(gnb1_data.mean() if len(gnb1_data) > 0 else np.nan)
        gnb1_sems.append(gnb1_data.sem() if len(gnb1_data) > 0 else 0)
    
    # Plot lines (WT black, GNB1 red)
    ax.errorbar(range(len(isis)), wt_means, yerr=wt_sems, color='black', marker='o',
                markersize=3, linewidth=1, label='WT', capsize=2, capthick=0.5)
    ax.errorbar(range(len(isis)), gnb1_means, yerr=gnb1_sems, color='red', marker='o',
                markersize=3, linewidth=1, label='I80T/+', capsize=2, capthick=0.5)
    
    # Format axes
    ax.set_xticks(range(len(isis)))
    ax.set_xticklabels([str(isi) for isi in isis])
    ax.set_xlabel('ISI (ms)', fontsize=8)
    ax.set_ylabel('EPSP Amplitude (mV)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='best', fontsize=7)
    
    # --- Load significance markers from FDR-corrected file ---
    df_markers = load_figure_5_significance_markers()
    annotate_with_sig_markers(ax, df_markers, 'Gabazine_Amplitude', pathway_name, 'WT_vs_GNB1', range(len(isis)))

def plot_10ms_ISI_breakdown(ax, df_traces, genotype, pathway_label):
    """
    Plot 10ms ISI breakdown traces for Figure 6 A/B.
    Shows: Control ± SEM (black), Gabazine ± SEM (magenta), Expected ± SEM (gray).
    A supralinearity shaded region highlights where Gabazine exceeds Expected.
    """
    isi = 10
    pathway_map = {
        'ECIII Input': 'perforant',
        'CA3 Apical Input': 'schaffer',
        'CA3 Basal Input': 'basal'
    }
    pathway_key = pathway_map.get(pathway_label, 'perforant')

    # genotype filter
    if genotype in ('GNB1', 'I80T/+'):
        genotype_targets = ['GNB1', 'I80T/+']
    else:
        genotype_targets = ['WT']

    condition = df_traces[
        (df_traces['Genotype'].isin(genotype_targets)) &
        (df_traces['ISI'] == isi)
    ].copy()

    if pathway_key == 'basal':
        subset = condition[condition['Pathway'] == 'Basal_Stratum_Oriens']
    elif pathway_key == 'perforant':
        subset = condition[condition['Channel'] == 'channel_1']
        if 'Pathway' in subset.columns:
            subset = subset[subset['Pathway'] != 'Basal_Stratum_Oriens']
    else:  # schaffer
        subset = condition[condition['Channel'] == 'channel_2']

    if len(subset) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.axis('off')
        return 0

    # ── helper: compute mean ± SEM across cells ──────────────────────────────
    def compute_mean_sem(df_sub, col):
        from collections import Counter
        traces = [r[col] for _, r in df_sub.iterrows()
                  if isinstance(r.get(col), np.ndarray) and len(r[col]) > 0]
        if not traces:
            return None, None
        target_len = Counter(len(t) for t in traces).most_common(1)[0][0]
        arr = np.array([t for t in traces if len(t) == target_len])
        return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    gabazine, gab_sem  = compute_mean_sem(subset, 'Gabazine_Trace')
    expected, exp_sem  = compute_mean_sem(subset, 'Expected_EPSP_Trace')

    if gabazine is None or expected is None:
        ax.axis('off')
        return 0

    # ── baseline-subtract ────────────────────────────────────────────────────
    bl = 100
    gabazine = gabazine - np.mean(gabazine[:bl])
    expected = expected - np.mean(expected[:bl])

    # ── trim to 300 ms @ 20 kHz ─────────────────────────────────────────────
    display_samples = 6000
    gabazine  = gabazine[:display_samples];  gab_sem  = gab_sem[:display_samples]
    expected  = expected[:display_samples];  exp_sem  = exp_sem[:display_samples]

    time = np.arange(display_samples) * 1000 / 20000   # ms
    n_cells = subset['Cell_ID'].nunique()

    # ── Supralinearity trace: Gabazine − Expected, in blue ──────────────────
    supra     = gabazine - expected
    supra_sem = np.sqrt(gab_sem**2 + exp_sem**2)   # propagate SEM

    # ── 1. Expected (gray) ──────────────────────────────────────────────────
    ax.fill_between(time, expected - exp_sem, expected + exp_sem,
                    color='gray', alpha=0.25, edgecolor='none')
    ax.plot(time, expected, color='gray', linewidth=0.8,
            label='Expected - Linear Summation')

    # ── 2. Gabazine / No-inhibition (magenta) ───────────────────────────────
    ax.fill_between(time, gabazine - gab_sem, gabazine + gab_sem,
                    color='magenta', alpha=0.20, edgecolor='none')
    ax.plot(time, gabazine, color='magenta', linewidth=1.0,
            label='Measured - No Inhibition')

    # ── 3. Supralinearity (blue): Measured No Inhib − Expected ──────────────
    ax.fill_between(time, supra - supra_sem, supra + supra_sem,
                    color='steelblue', alpha=0.25, edgecolor='none')
    ax.plot(time, supra, color='steelblue', linewidth=1.0,
            label='Supralinearity (No Inhib − Expected)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.set_title(pathway_label, fontsize=8, fontweight='bold')
    ax.axis('off')

    # Scale bar on leftmost panel only
    if pathway_label == 'ECIII Input':
        add_scale_bar(ax, 50, 2, x_pos=0.8, y_pos=0.2)

    return n_cells



def plot_example_ISI_trace(ax, df_traces, df_amplitudes, isi, pathway_label, annotate=False):
    """
    Plot a single example cell trace (Control vs Gabazine) for Figure 5 Row 1.

    GABAB area = ALL area where gabazine trace is below zero (baseline).
    fill_between(time, gabazine, 0, where=gabazine<0) fills the region
    BETWEEN the gabazine trace and y=0 — this is the correct GABAB integral.
    """
    # ----- HARDCODED EXAMPLE CELL (chosen for visual quality) -----
    #20240905_c3
    PREFERRED_CELL = {
        'perforant': '20240905_c3',
        'schaffer':  '20240905_c3',
        'basal':     None,           # fall back to auto-select for basal
    }

    pathway_map = {
        'ECIII Input':       'perforant',
        'CA3 Apical Input':  'schaffer',
        'CA3 Basal Input':   'basal',
    }
    pathway_key  = pathway_map.get(pathway_label, 'perforant')
    pathway_name = {
        'perforant': 'Perforant',
        'schaffer':  'Schaffer',
        'basal':     'Basal_Stratum_Oriens',
    }[pathway_key]

    # ----- resolve example cell -----
    best_cell = PREFERRED_CELL.get(pathway_key)

    if best_cell is None:
        # auto-select: best combined score (GABAB area + gabazine peak)
        sub_amp = df_amplitudes[
            (df_amplitudes['ISI']      == isi)         &
            (df_amplitudes['Pathway']  == pathway_name) &
            df_amplitudes['Genotype'].isin(['WT', 'GNB1'])
        ].copy()
        if len(sub_amp) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.axis('off')
            return
        gab_peak  = sub_amp['Gabazine_Amplitude'].fillna(0)
        gabab_neg = sub_amp['GABAB_Area'].fillna(0).abs()
        norm_peak = (gab_peak  - gab_peak.min())  / (gab_peak.max()  - gab_peak.min()  + 1e-9)
        norm_neg  = (gabab_neg - gabab_neg.min()) / (gabab_neg.max() - gabab_neg.min() + 1e-9)
        sub_amp['_score'] = norm_peak + 2.0 * norm_neg
        best_cell = sub_amp.sort_values('_score', ascending=False).iloc[0]['Cell_ID']

    # ----- fetch trace -----
    sub_trace = df_traces[(df_traces['Cell_ID'] == best_cell) & (df_traces['ISI'] == isi)]
    if pathway_key == 'basal':
        sub_trace = sub_trace[sub_trace['Pathway'] == 'Basal_Stratum_Oriens']
    elif pathway_key == 'perforant':
        sub_trace = sub_trace[sub_trace['Channel'] == 'channel_1']
        if 'Pathway' in sub_trace.columns:
            sub_trace = sub_trace[sub_trace['Pathway'] != 'Basal_Stratum_Oriens']
    else:
        sub_trace = sub_trace[sub_trace['Channel'] == 'channel_2']

    if len(sub_trace) == 0:
        ax.text(0.5, 0.5, f'No trace\n({best_cell})', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.axis('off')
        return

    control  = sub_trace.iloc[0].get('Control_Trace')
    gabazine = sub_trace.iloc[0].get('Gabazine_Trace')

    if not isinstance(control, np.ndarray) or not isinstance(gabazine, np.ndarray):
        ax.axis('off')
        return

    # ----- baseline-subtract (first 100 samples = 5 ms pre-stim) -----
    bl = 100
    control  = control  - np.mean(control [:bl])
    gabazine = gabazine - np.mean(gabazine[:bl])

    # ----- trim display: 500 ms for 50-ms ISI, 350 ms for 10-ms ISI -----
    display_ms      = 550 if isi == 50 else 350
    display_samples = int(display_ms * 20000 / 1000)
    control  = control [:display_samples]
    gabazine = gabazine[:display_samples]
    time     = np.arange(len(control)) * 1000 / 20000    # → ms

    # ----- plot traces -----
    ax.plot(time, gabazine, color='magenta', linewidth=1.0, label='Measured - No Inhibition')
    ax.plot(time, control,  color='black',   linewidth=1.0, label='Measured - With Inhibition')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    # ----- GABAB shading: fill BETWEEN gabazine trace and y=0, where gabazine < 0 -----
    # This is the integral of all negative-going area (the slow IPSP)
    neg_mask = gabazine < 0
    if np.any(neg_mask):
        ax.fill_between(time, gabazine, 0,
                        where=neg_mask,
                        color='gray', alpha=0.35, edgecolor='none')

    # ----- annotations (only on annotated panel) -----
    if annotate:
        # Find the LAST local peak of the gabazine train (=summed/final response).
        # For a multi-pulse train, we look in the second half of the train window.
        half = len(gabazine) // 2
        last_half = gabazine[half:]
        last_peak_local = int(np.nanargmax(last_half))
        gab_peak_idx  = half + last_peak_local
        gab_peak_val  = float(gabazine[gab_peak_idx])
        ctrl_peak_val = float(control [gab_peak_idx])

        # x-coordinate of the peak (ms) + small offset for arrows
        peak_t = float(time[gab_peak_idx])

        # --- Arrow 1: Excitation (magenta) — baseline → gabazine peak ---
        ax.annotate('', xy=(peak_t, 0), xytext=(peak_t, gab_peak_val),
                    arrowprops=dict(arrowstyle='<->', color='magenta',
                                   lw=1.0, mutation_scale=8))
        ax.text(peak_t + time[-1]*0.03, gab_peak_val * 0.5,
                'Excitation', color='magenta', fontsize=7,
                va='center', fontweight='bold')

        # --- Arrow 2: Inh(GABAA) (black) — control peak → gabazine peak ---
        if ctrl_peak_val < gab_peak_val - 0.2:
            ax.annotate('', xy=(peak_t, ctrl_peak_val), xytext=(peak_t, gab_peak_val),
                        arrowprops=dict(arrowstyle='<->', color='black',
                                       lw=1.0, mutation_scale=8))
            ax.text(peak_t + time[-1]*0.03,
                    (gab_peak_val + ctrl_peak_val) * 0.5,
                    'Inh(GABAA)', color='black', fontsize=7,
                    va='center', fontweight='bold')

        # --- GABAB label inside the shaded area ---
        if np.any(neg_mask):
            deepest_idx = int(np.nanargmin(gabazine))
            deepest_t   = float(time[deepest_idx])
            deepest_v   = float(gabazine[deepest_idx])
            # place label at the deepest point, halfway between trace and zero
            ax.text(deepest_t, deepest_v * 0.55,
                    'Slow IPSP Area\n(GABAB)',
                    ha='center', va='center', fontsize=7,
                    color='dimgray', fontweight='bold')

    ax.axis('off')
    add_scale_bar(ax, 50, 2, x_pos=0.55, y_pos=0.15)



def get_n_count_string(ns_train, n_300):
    """Formats N-counts as a range for trains and a single value for 300ms."""
    if not ns_train:
        return f"n={n_300}"
    
    min_n = min(ns_train)
    max_n = max(ns_train)
    
    if min_n == max_n:
        range_str = f"n={min_n}"
    else:
        range_str = f"n={min_n}-{max_n}"
    
    if n_300 != min_n or n_300 != max_n:
        return f"{range_str}; 300ms n={n_300}"
    return range_str

def plot_metric_comparison(ax, df_amplitudes, pathway_name, column, ylabel, add_legend=False, df_stats=None, panel_id=None):
    """
    Plot line comparison of WT vs GNB1 for a given metric.
    WT = black, GNB1 = red
    """
    # ISI values
    isis = [300, 100, 50, 25, 10]
    
    # Collect data for WT and GNB1
    wt_means, wt_sems, wt_ns = [], [], []
    gnb1_means, gnb1_sems, gnb1_ns = [], [], []
    
    for isi in isis:
        # WT data
        wt_subset = df_amplitudes[
            (df_amplitudes['Genotype'] == 'WT') &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ][column].dropna()
        wt_means.append(wt_subset.mean() if len(wt_subset) > 0 else np.nan)
        wt_sems.append(wt_subset.sem() if len(wt_subset) > 0 else 0)
        # N-count logic: unique Cell_ID count
        wt_n = df_amplitudes[
            (df_amplitudes['Genotype'] == 'WT') &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ]['Cell_ID'].dropna().nunique()
        wt_ns.append(wt_n)
        
        # GNB1 data
        gnb1_subset = df_amplitudes[
            (df_amplitudes['Genotype'].isin(['GNB1', 'I80T/+'])) &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ][column].dropna()
        gnb1_means.append(gnb1_subset.mean() if len(gnb1_subset) > 0 else np.nan)
        gnb1_sems.append(gnb1_subset.sem() if len(gnb1_subset) > 0 else 0)
        gnb1_n = df_amplitudes[
            (df_amplitudes['Genotype'].isin(['GNB1', 'I80T/+'])) &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ]['Cell_ID'].dropna().nunique()
        gnb1_ns.append(gnb1_n)
    
    # Format labels with N-ranges
    wt_label = f"WT ({get_n_count_string(wt_ns[1:], wt_ns[0])})"
    gnb1_label = f"I80T/+ ({get_n_count_string(gnb1_ns[1:], gnb1_ns[0])})"

    # Plot lines with error bands (WT black, GNB1 red)
    ax.errorbar(range(len(isis)), wt_means, yerr=wt_sems, color='black', marker='o',
                markersize=2.5, linewidth=1.0, label=wt_label, capsize=2, capthick=0.5)
    ax.errorbar(range(len(isis)), gnb1_means, yerr=gnb1_sems, color='red', marker='o',
                markersize=2.5, linewidth=1.0, label=gnb1_label, capsize=2, capthick=0.5)
    
    # Statistical annotations (Stars for post-hoc, Hashtag for main effects)
    if df_stats is not None:
        # Map column names to comparison strings in markers CSV
        comp_map = {
            'Gabazine_Amplitude': 'Gabazine_Amplitude',
            'Estimated_Inhibition_Amplitude': 'Inhibition_Amplitude',
            'E_I_Imbalance': 'E_I_Imbalance',
            'Gabazine_Supralinearity': 'Gabazine_Supralinearity',
            'GABAB_Area': 'GABAB_Area'
        }
        analysis_key = comp_map.get(column, column)
        
        # annotate_with_sig_markers(ax, markers_df, analysis, pathway, comparison, x_coords)
        # x_coords for line plots are range(len(isis))
        annotate_with_sig_markers(ax, df_stats, analysis_key, pathway_name, 'WT_vs_GNB1', range(len(isis)))

    # Format axes
    ax.set_xticks(range(len(isis)))
    ax.set_xticklabels([str(isi) for isi in isis])
    ax.set_xlabel('ISI (ms)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Line plots: use matplotlib auto-scaling (no clean_yticks — avoids decimal tick values)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    
    if np.any(np.array(wt_means) < 0) or np.any(np.array(gnb1_means) < 0):
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    
    if add_legend:
        ax.legend(frameon=False, loc='best', fontsize=6)
        
    return ax.get_ylim()

def plot_supplemental_figure_1_helper(isi_order, metrics, pathways, genotypes, df, axes):
    for row_idx, genotype in enumerate(genotypes):
        for col_idx, (path_label, path_code) in enumerate(pathways):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this panel (Genotype + Pathway)
            subset = df[(df['Pathway'] == path_code) & (df['Genotype'] == genotype)]
            
            if subset.empty:
                continue
            
            # Plot standard metrics from the list
            for metric, label, color in metrics:
                # Group by ISI
                grouped = subset.groupby('ISI')[metric].agg(['mean', 'sem', 'count']).reindex(isi_order)
                
                # Check if data exists for this metric
                if grouped['mean'].notna().any():
                    # Plot
                    ax.errorbar(range(len(isi_order)), grouped['mean'], yerr=grouped['sem'],
                               label=None, color=color, marker='o', markersize=3, capsize=3,
                               linewidth=1.5 if metric != 'Expected_EPSP_Amplitude' else 1.0,
                               linestyle='-' if metric != 'Expected_EPSP_Amplitude' else '-')
                    
            # Add horizontal line at zero
            ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

            # Styling
            ax.set_xticks(range(len(isi_order)))
            ax.set_xticklabels(isi_order)
            
            if row_idx == 1:
                ax.set_xlabel('ISI (ms)')
            if col_idx == 0:
                ax.set_ylabel('Amplitude (mV)')
                
            title = f"{genotype} - {path_label}"
            ax.set_title(title, fontsize=10)

# ----------------------------------------------------------------------------------
# Noise and Artifact Removal Functions
# ----------------------------------------------------------------------------------

def remove_noise(data, noise_times, acquisition_frequency, delete_noise_duration_list):
    """Removes noise artifacts from data.

    Parameters
    ----------
    data : np.array
        The data containing noise artifacts.
    noise_times : list or int or float
        A list of noise times or a single noise time in milliseconds.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    delete_noise_duration_list : list or int or float
        A list of durations or a single duration to delete around each noise time in milliseconds.

    Returns
    -------
    np.array
        The data with noise artifacts removed and interpolated.
    """
    processed_data = np.copy(data)

    noise_times = [noise_times] if isinstance(noise_times, (int, float)) else noise_times
    delete_noise_duration_list = [delete_noise_duration_list] if isinstance(delete_noise_duration_list, (
    int, float)) else delete_noise_duration_list

    if len(noise_times) != len(delete_noise_duration_list):
        raise ValueError("noise_times and delete_noise_duration_list must have the same length.")

    for noise_time, delete_noise_duration in zip(noise_times, delete_noise_duration_list):
        current_noise_index = int(noise_time * acquisition_frequency / 1000)
        delete_start_index = max(0, current_noise_index - int(delete_noise_duration * acquisition_frequency / 1000))
        delete_end_index = min(len(processed_data),
                               current_noise_index + int(delete_noise_duration * acquisition_frequency / 1000))

        processed_data[delete_start_index:delete_end_index] = np.nan

        not_nan_indices = np.arange(0, len(processed_data))[~np.isnan(processed_data)]
        processed_data = np.interp(np.arange(0, len(processed_data)), not_nan_indices,
                                   processed_data[~np.isnan(processed_data)])

    return processed_data

def normalize_plateau_trace(trace, baseline_window=1000):
    """Normalize a trace by subtracting the baseline (first baseline_window samples)."""
    baseline = np.mean(trace[:baseline_window])
    return trace - baseline

def remove_artifacts_automated(data, acquisition_frequency, delete_start_stim, delete_end_stim):
    """Removes stimulation artifacts from data automatically without knowing the stimulation times.

    Parameters
    ----------
    data : np.array
        The data containing stimulation artifacts.
    acquisition_frequency : int or float
        The acquisition frequency of the data in Hz.
    delete_start_stim : int or float
        The time in milliseconds to start deleting before the stimulation artifact.
    delete_end_stim : int or float
        The time in milliseconds to stop deleting after the stimulation artifact.

    Returns
    -------
    np.array
        The data with artifacts removed and interpolated.
    """
    from scipy.signal import find_peaks
    
    processed_data = np.copy(data)  # Make a copy of the data to avoid modifying the original

    #first get the derivative of the data
    derivative = np.diff(processed_data, n=1)
    #find all possible APs first
    AP_peaks = find_peaks(derivative, height=0)[0]
    #Then find the negative inflection points
    negative_inflection_points = find_peaks(-derivative, height=0.2)[0] #0.2 is a good threshold for negative inflection points
    #If a peak occurs 1ms from an AP_peak, remove it from all peaks
    for peak in negative_inflection_points:
        for AP_peak in AP_peaks:
            if abs(peak - AP_peak) < acquisition_frequency / 1000:
                negative_inflection_points = negative_inflection_points[negative_inflection_points != AP_peak]
    
    for inflection in negative_inflection_points:
        baseline_start = int(inflection * acquisition_frequency/1000) - int(0.5 * acquisition_frequency/1000)
        baseline_voltage = np.mean(processed_data[baseline_start:int(inflection * acquisition_frequency/1000)])
        delete_start_index = max(0, inflection - int(delete_start_stim * acquisition_frequency / 1000))
        interp_start_index = min(len(processed_data), inflection - int(delete_start_stim * acquisition_frequency / 1000))
        delete_end_index = min(len(processed_data), inflection + int(delete_end_stim * acquisition_frequency / 1000))
        if any(processed_data[delete_end_index:] > baseline_voltage ):
                interp_end_index = np.where(processed_data[delete_end_index:] > baseline_voltage)[0][0] + delete_end_index
        else:
            interp_end_index = delete_end_index
        current_window = np.arange(interp_start_index, interp_end_index)
        processed_data[current_window] = np.nan
        processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])
    
    return processed_data


# =========================================================================
# Figure 7 Traces 
# =========================================================================
def plot_traces_GIRK_exp(ax, traces_before, traces_after, genotype, drug_name, add_legend=False, add_scale=False):
    """Plot Gabazine vs Gabazine+Drug traces with SEM"""
    
    # Define colors locally
    colors = {'WT': 'k', 'GNB1': 'r', 'I80T/+': 'r'}
    acq_freq = 20000
    
    color = colors[genotype]
    drug_color = 'darkorange'
    
    before_list = []
    after_list = []
    
    for cell_id, cell_data in traces_before.items():
        if cell_data['genotype'] == genotype:
            if 'Both' in cell_data['traces']:
                trace = cell_data['traces']['Both']
                if len(trace) >= 30000:
                    before_list.append(trace[10000:30000])
    
    for cell_id, cell_data in traces_after.items():
        if cell_data['genotype'] == genotype:
            if 'Both' in cell_data['traces']:
                trace = cell_data['traces']['Both']
                if len(trace) >= 30000:
                    after_list.append(trace[10000:30000])
    
    if before_list and after_list:
        before_mean = np.mean(before_list, axis=0)
        before_sem = np.std(before_list, axis=0) / np.sqrt(len(before_list))
        after_mean = np.mean(after_list, axis=0)
        after_sem = np.std(after_list, axis=0) / np.sqrt(len(after_list))
        
        # Downsample for SVG
        ds = 20
        before_mean = before_mean[::ds]
        before_sem = before_sem[::ds]
        after_mean = after_mean[::ds]
        after_sem = after_sem[::ds]
        time = np.arange(len(before_mean)) * ds / acq_freq  # Time in seconds
        
        # Plot Before Drug
        ax.fill_between(time, before_mean - before_sem, before_mean + before_sem, 
                        color=color, alpha=0.3)
        ax.plot(time, before_mean, color=color, linewidth=1.2, label='Before Drug')
        
        # Plot After Drug
        ax.fill_between(time, after_mean - after_sem, after_mean + after_sem, 
                        color=drug_color, alpha=0.3)
        ax.plot(time, after_mean, color=drug_color, linewidth=1.2, 
                label='After Drug')
        
        if add_scale:
            # Draw scale bar inline (time in seconds, voltage in mV)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_scale = 0.2  # 200 ms = 0.2 seconds
            y_scale = 10   # 10 mV
            
            x_start = xlim[0] + 0.85 * (xlim[1] - xlim[0])
            y_start = ylim[0] + 0.15 * (ylim[1] - ylim[0])
            
            # Horizontal bar
            ax.plot([x_start, x_start + x_scale], [y_start, y_start], 'k-', linewidth=1.5)
            # Vertical bar
            ax.plot([x_start, x_start], [y_start, y_start + y_scale], 'k-', linewidth=1.5)
            
            # Labels
            ax.text(x_start + x_scale/2, y_start - (ylim[1]-ylim[0])*0.03, '200 ms', 
                   ha='center', va='top', fontsize=7)
            ax.text(x_start - (xlim[1]-xlim[0])*0.02, y_start + y_scale/2, '10 mV', 
                   ha='right', va='center', rotation=90, fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No traces', ha='center', va='center', 
                transform=ax.transAxes, fontsize=8, color='gray')
    
    ax.set_title(genotype, fontweight='bold', fontsize=9)
    if add_legend:
        ax.legend(frameon=False, fontsize=7, loc='upper right')
    
    # Turn off axis AFTER plotting so autoscale works correctly
    ax.axis('off')

def plot_GIRK_bars(ax, df_before, df_after, drug_name, sig_within, sig_between):
    # EXACT SAME COLORS AS plot_bar_scatter_fig2
    colors = {'WT': 'black', 'GNB1': 'red', 'I80T/+': 'red'}
    
    genotypes = ['WT', 'I80T/+']
    x_pos = 0
    positions = {}
    data_by_group = {}
    bar_width = 0.6  # Same as plot_bar_scatter
    group_spacing = 2.0
        
    for genotype in genotypes:
        color = colors[genotype]
            
        before = df_before[df_before['Genotype'] == genotype]
        after = df_after[df_after['Genotype'] == genotype]
            
        paired = before.merge(after[['Cell_ID', 'Plateau_Area']], 
                              on='Cell_ID', suffixes=('_before', '_after'))
            
        if len(paired) == 0:
            x_pos += group_spacing
            continue
            
        control_mean = paired['Plateau_Area_before'].mean()
        paired['Norm_Before'] = paired['Plateau_Area_before'] / control_mean
        paired['Norm_After'] = paired['Plateau_Area_after'] / control_mean
        
        n = len(paired)
        x_ctrl = x_pos
        x_drug = x_pos + 0.9
        
        positions[genotype] = x_drug
        data_by_group[genotype] = paired['Norm_After']
        
        mean_ctrl = paired['Norm_Before'].mean()
        mean_drug = paired['Norm_After'].mean()
        sem_ctrl = paired['Norm_Before'].std() / np.sqrt(n)
        sem_drug = paired['Norm_After'].std() / np.sqrt(n)
        
        # EXACT SAME STYLING AS plot_bar_scatter
        ax.bar(x_ctrl, mean_ctrl, bar_width, color=color, alpha=0.5)
        ax.bar(x_drug, mean_drug, bar_width, color=color, alpha=0.5)
        ax.errorbar(x_ctrl, mean_ctrl, yerr=sem_ctrl, fmt='o', color=color, capsize=1, elinewidth=1, markersize=2)
        ax.errorbar(x_drug, mean_drug, yerr=sem_drug, fmt='o', color=color, capsize=1, elinewidth=1, markersize=2)
        
        ax.scatter([x_ctrl]*n, paired['Norm_Before'], color=color, s=2, zorder=3)
        ax.scatter([x_drug]*n, paired['Norm_After'], color=color, s=2, zorder=3)
        
        for _, row in paired.iterrows():
            ax.plot([x_ctrl, x_drug], [row['Norm_Before'], row['Norm_After']], 
                    color='gray', alpha=0.3, linewidth=0.5)
        
        # Within-genotype bracket
        ymax = max(paired['Norm_Before'].max(), paired['Norm_After'].max()) + 0.1
        ax.plot([x_ctrl, x_ctrl, x_drug, x_drug], [ymax, ymax+0.05, ymax+0.05, ymax], 'k-', lw=0.8)
        sig = sig_within.get(genotype, '')
        ax.text((x_ctrl + x_drug)/2, ymax + 0.06, sig, ha='center', fontsize=7, fontweight='bold')
        
        x_pos += group_spacing
    
    # Between-genotype bracket
    if 'WT' in positions and ('GNB1' in positions or 'I80T/+' in positions):
        x1, x2 = positions['WT'], positions.get('GNB1', positions.get('I80T/+', 1))
        y1 = data_by_group['WT'].max() if len(data_by_group['WT']) > 0 else 1
        y2 = data_by_group.get('GNB1', data_by_group.get('I80T/+', [])).max() if len(data_by_group.get('GNB1', data_by_group.get('I80T/+', []))) > 0 else 1
        ymax = max(y1, y2) + 0.45
        ax.plot([x1, x1, x2, x2], [ymax, ymax+0.05, ymax+0.05, ymax], 'k-', lw=0.8)
        ax.text((x1 + x2)/2, ymax + 0.06, sig_between, ha='center', fontsize=7, fontweight='bold')
    
    ax.set_xticks([0, 0.9, 2.0, 2.9])
    ax.set_xticklabels(['WT\nBefore', 'WT\nAfter', 'GNB1\nBefore', 'GNB1\nAfter'], fontsize=7)
    ax.set_ylabel('Normalized\nPlateau Area', fontsize=8)
    ax.set_title(drug_name, fontsize=9, fontweight='bold')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ==================================================================================================
# FIGURE 5: GABAb ANALYSIS HELPER FUNCTIONS
# ==================================================================================================

def plot_gabab_traces(ax, gabab_traces, pathway_key, title, label, gabab_metrics_df=None):
    """
    Plot GABAb traces for a given pathway in Figure 5.
    
    Parameters:
        ax: matplotlib axes
        gabab_traces: dict with structure {condition: {cell_id: {genotype, sex, traces}}}
        pathway_key: str, pathway name ('Perforant Path', 'Schaffer Collateral', etc.)
        title: str, subplot title
        label: str, subplot label (e.g., 'A', 'E')
        gabab_metrics_df: DataFrame, filtered metrics (only cells with valid analysis)
    """
    add_subplot_label(ax, label)
    ax.set_title(title, fontweight='bold', fontsize=8)
    ax.axis('off')
    
    if 'gabazine' not in gabab_traces:
        return
    
    # CRITICAL FIX: Get list of cells with valid metrics for this pathway
    # This ensures N in trace panels (A, B, C) matches N in quantification panels (D, E, F)
    valid_cells = set()
    if gabab_metrics_df is not None:
        pathway_metrics = gabab_metrics_df[gabab_metrics_df['Channel_Name'] == pathway_key]
        valid_cells = set(pathway_metrics['Cell_ID'].values)
        
    wt_traces = []
    gnb1_traces = []
    
    for cell_id, cell_data in gabab_traces['gabazine'].items():
        # Skip cells without valid metrics if metrics df provided
        if gabab_metrics_df is not None and cell_id not in valid_cells:
            continue
            
        if not isinstance(cell_data, dict) or 'traces' not in cell_data:
            continue
            
        genotype = cell_data.get('genotype', '')
        traces_dict = cell_data['traces']
        
        if pathway_key in traces_dict:
            trace = traces_dict[pathway_key]
            # Only include traces with > 4000 samples to avoid short/truncated traces
            if isinstance(trace, np.ndarray) and len(trace) > 4000:
                if genotype == 'WT':
                    wt_traces.append(trace[:6000])
                elif genotype in ('GNB1', 'I80T/+'):
                    gnb1_traces.append(trace[:6000])
    
    if wt_traces and gnb1_traces:
        # Truncate all traces to minimum length
        min_len = min(min(len(t) for t in wt_traces), min(len(t) for t in gnb1_traces))
        wt_traces = [t[:min_len] for t in wt_traces]
        gnb1_traces = [t[:min_len] for t in gnb1_traces]
        
        wt_mean = np.mean(wt_traces, axis=0)
        gnb1_mean = np.mean(gnb1_traces, axis=0)
        wt_sem = np.std(wt_traces, axis=0) / np.sqrt(len(wt_traces))
        gnb1_sem = np.std(gnb1_traces, axis=0) / np.sqrt(len(gnb1_traces))
        
        time = np.arange(len(wt_mean)) / 20  # 20 kHz
        
        ax.fill_between(time, wt_mean - wt_sem, wt_mean + wt_sem, color='black', alpha=0.3, edgecolor='none')
        ax.plot(time, wt_mean, 'k-', linewidth=1, label=f'WT (n={len(wt_traces)})')
        
        ax.fill_between(time, gnb1_mean - gnb1_sem, gnb1_mean + gnb1_sem, color='red', alpha=0.3, edgecolor='none')
        ax.plot(time, gnb1_mean, 'r-', linewidth=1, label=f'GNB1 (n={len(gnb1_traces)})')
        
        add_scale_bar(ax, 50, 1, x_pos=0.85, y_pos=0.15)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    elif wt_traces or gnb1_traces:
        # Only one genotype available
        traces_to_plot = wt_traces if wt_traces else gnb1_traces
        geno = 'WT' if wt_traces else 'I80T/+'
        color = 'black' if geno == 'WT' else 'red'
        
        min_len = min(len(t) for t in traces_to_plot)
        traces_to_plot = [t[:min_len] for t in traces_to_plot]
        
        mean_trace = np.mean(traces_to_plot, axis=0)
        sem_trace = np.std(traces_to_plot, axis=0) / np.sqrt(len(traces_to_plot))
        time = np.arange(len(mean_trace)) / 20
        
        ax.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.3, edgecolor='none')
        ax.plot(time, mean_trace, color=color, linewidth=1, label=f'{geno} (n={len(traces_to_plot)})')
        
        add_scale_bar(ax, 50, 1, x_pos=0.85, y_pos=0.15)
        ax.legend(frameon=False, fontsize=8, loc='upper right')


def plot_gabab_metric_bar(ax, gabab_df, pathway_key, metric_col, ylabel, label, df_stats=None, pathway_match=None):
    """
    Plot a single metric bar plot for GABAb analysis.
    
    Parameters:
        ax: matplotlib axes
        gabab_df: DataFrame with GABAb metrics
        pathway_key: str, pathway name for filtering
        metric_col: str, column name for the metric
        ylabel: str, y-axis label
        label: str, subplot label (e.g., 'B', 'F')
        df_stats: DataFrame with statistical results (optional)
        pathway_match: str, pathway name for matching in stats (optional)
    """
    add_subplot_label(ax, label)
    
    pathway_data = gabab_df[gabab_df['Channel_Name'] == pathway_key].copy()
    
    if pathway_data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Use plot_bar_scatter
    plot_bar_scatter(ax, pathway_data, x_col='Genotype', y_col=metric_col, 
                    hue_col='Genotype', order=['WT', 'I80T/+'])
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(1)
    
    # Add statistical annotation
    if df_stats is not None and pathway_match is not None:
        panel_id = f"Fig 5{label}"
        annotate_from_stats(ax, df_stats, panel_id, pathway_match, x1=0, x2=1, y_pos=ax.get_ylim()[1] * 0.95)


def plot_gabab_vm_change(ax, vm_csv_path, label, df_stats=None):
    """
    Plot voltage change bar plot for Baclofen effects.
    
    Parameters:
        ax: matplotlib axes
        vm_csv_path: str, path to Baclofen_Vm_Change.csv
        label: str, subplot label
        df_stats: pd.DataFrame, optional stats results
    """
    add_subplot_label(ax, label)
    
    if os.path.exists(vm_csv_path):
        df_vm = pd.read_csv(vm_csv_path)
        
        # Rename GNB1 -> I80T/+ for display consistency
        df_vm = rename_genotype(df_vm)
        
        # Enforce negative values (Hyperpolarization = Drop from Zero)
        # If values are positive (magnitude), flip them.
        # If values are already negative, keep them.
        # Logic: We assume Baclofen HYPERPOLARIZES, so Vm Change should be negative.
        # Some datasets store this as absolute magnitude. We want "negative change".
        df_vm['Voltage Change'] = -1 * df_vm['Voltage Change'].abs()
        
        plot_bar_scatter(ax, df_vm, 'Genotype', 'Voltage Change', 'Genotype', order=['WT', 'I80T/+'])
        ax.set_ylabel('ΔVm (mV)')
        ax.set_title('Resting Potential Change', fontsize=8, fontweight='bold')
        ax.set_box_aspect(1)
        
        # Add statistical annotation from Figure 7 stats
        if df_stats is not None and 'Drug' in df_stats.columns:
            bac_row = df_stats[(df_stats['Drug'] == 'Baclofen') &
                               (df_stats['Comparison'].str.contains('ΔVm', na=False))]
            if not bac_row.empty:
                sig = bac_row.iloc[0].get('Significance', 'ns')
                wt_vals   = df_vm[df_vm['Genotype'] == 'WT']['Voltage Change'].dropna()
                gnb1_vals = df_vm[df_vm['Genotype'] == 'I80T/+']['Voltage Change'].dropna()
                if len(wt_vals) > 0 and len(gnb1_vals) > 0:
                    # Values are negative (hyperpolarisation) — annotate below bars
                    y_min = min(wt_vals.mean(), gnb1_vals.mean())
                    y_pos = y_min * 1.20   # 20% below the most negative mean
                    ax.text(0.5, y_pos, sig, ha='center', va='top', fontsize=10,
                            fontweight='bold')
                    if sig != 'ns':
                        ax.plot([0, 1], [y_pos + 0.5, y_pos + 0.5], 'k-', lw=0.8)
    else:
        ax.text(0.5, 0.5, 'No Vm Data', ha='center', va='center')
        ax.axis('off')



def plot_baclofen_vm_traces(ax, vm_traces_path, label='H'):
    """
    Plot Vm traces for WT and GNB1 stacked vertically on a single axis.
    Overlay Gabazine vs Gabazine + Baclofen for each to show resting Vm change.
    
    Parameters:
        ax: matplotlib axes (single)
        vm_traces_path: str, path to Baclofen_Vm_Example_Traces.pkl
        label: str, subplot label
    """
    import os
    import numpy as np
    import pandas as pd
    
    add_subplot_label(ax, label)
    
    if not os.path.exists(vm_traces_path):
        ax.text(0.5, 0.5, 'No Vm Trace Data', ha='center', va='center', fontsize=8)
        ax.axis('off')
        return
    
    vm_data = pd.read_pickle(vm_traces_path)
    # Load reported Vm changes to offset the traces (reconstruct un-clamped Vm)
    vm_change_df = pd.read_csv('paper_data/gabab_analysis/Baclofen_Vm_Change.csv')
    
    acq_freq = 20000
    n_samples = 1000  # 50 ms at 20 kHz
    time_ms = np.arange(n_samples) / acq_freq * 1000  # 0-50 ms
    
    # User requested specific WT cell
    wt_id = '20260206_c3'
    wt_row = vm_change_df[vm_change_df['Cell_ID'] == wt_id]
    wt_delta = wt_row['Voltage Change'].values[0] if not wt_row.empty else 0

    # User requested specific GNB1 cell
    gnb1_id = '20251209_c2'
    gnb1_row = vm_change_df[vm_change_df['Cell_ID'] == gnb1_id]
    gnb1_delta = gnb1_row['Voltage Change'].values[0] if not gnb1_row.empty else 0

    # Add legend at the top left manually
    ax.plot([], [], color='black', label='Gabazine')
    ax.plot([], [], color='gray', label='Gabazine + Baclofen')
    ax.legend(frameon=False, loc='upper left', fontsize=7, bbox_to_anchor=(0.0, 1.25))

    # PLOT STACKED
    offset = 20 # mV shift for GNB1
    
    # Store processed traces for scale bar calc
    all_y = []
    
    for i, (cell_id, delta, geno, y_shift) in enumerate([(wt_id, wt_delta, 'WT', 0), (gnb1_id, gnb1_delta, 'GNB1', -offset)]):
        if cell_id and cell_id in vm_data:
            cdata = vm_data[cell_id]
            gab_trace = cdata['Gabazine'][:n_samples] + y_shift
            bac_trace = cdata['Gabazine + Baclofen'][:n_samples].copy() + y_shift
            
            # Reconstruction offset
            gab_mean_orig = np.mean(cdata['Gabazine'][:1000])
            bac_mean_orig = np.mean(cdata['Gabazine + Baclofen'][:1000])
            
            # We want baclofen to be lower by `delta`
            needed_bac_mean = gab_mean_orig - delta
            reconstruct_offset = needed_bac_mean - bac_mean_orig
            
            bac_trace += reconstruct_offset 
            
            # Plot
            base_color = 'black' if geno == 'WT' else 'red'
            bac_color = 'gray' if geno == 'WT' else 'salmon'
            
            ax.plot(time_ms, gab_trace, color=base_color, linewidth=0.8)
            ax.plot(time_ms, bac_trace, color=bac_color, linewidth=0.8)
            
            all_y.extend(gab_trace)
            all_y.extend(bac_trace)
            
            # Dashed baselines
            gab_base_plot = np.mean(gab_trace[:1000])
            bac_base_plot = np.mean(bac_trace[:1000])
            ax.plot([-5, 50], [gab_base_plot, gab_base_plot], color=base_color, linestyle='--', linewidth=0.5, alpha=0.5)
            ax.plot([-5, 50], [bac_base_plot, bac_base_plot], color=bac_color, linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Label Genotype
            display_geno = 'I80T/+' if geno == 'GNB1' else 'WT'
            ax.text(45, gab_base_plot + 1, display_geno, fontsize=8, fontweight='bold', color=base_color, ha='right')
            
            if geno == 'WT':
                # Delta Label and Arrow
                ax.text(-2, gab_base_plot, 'Resting\n$V_m$', fontsize=7, ha='right', va='center')
                # Arrow for delta Vm
                ax.annotate('', xy=(25, bac_base_plot), xytext=(25, gab_base_plot),
                            arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
                ax.text(27, (gab_base_plot + bac_base_plot)/2, '$\Delta V_m$', fontsize=7, va='center')

    # Add Scale Bar (L-SHAPE)
    if all_y:
        y_min_total = min(all_y)
        ax.set_xlim(-15, 60)
        overall_range = np.max(all_y) - y_min_total
        center = (np.max(all_y) + y_min_total)/2
        # Expand ylim to compress traces visually
        ax.set_ylim(center - overall_range * 1.5 - 5, center + overall_range * 1.5 + 5)
        
        ax.axis('off')
        
        # Position: Bottom Right (within 50 ms window)
        bar_x = 30
        bar_y = y_min_total - 5
        
        # 10 ms Horizontal
        ax.plot([bar_x, bar_x + 10], [bar_y, bar_y], 'k-', linewidth=1.5)
        ax.text(bar_x + 5, bar_y - 1, '10 ms', ha='center', va='top', fontsize=6)
        
        # 5 mV Vertical
        ax.plot([bar_x + 10, bar_x + 10], [bar_y, bar_y + 5], 'k-', linewidth=1.5)
        ax.text(bar_x + 11, bar_y + 2.5, '5 mV', ha='left', va='center', fontsize=6)
        
        # Set limits
        if all_y:
            ax.set_ylim(bar_y - 5, max(all_y) + 10)
    
    ax.axis('off')

def plot_fi_curve_gabab(ax, df, geno, title):
    """Plot FI curves for Baclofen data (Gabazine vs Gabazine+Baclofen)."""
    subset = df[df['Genotype'] == geno]
    if subset.empty:
        ax.text(0.5, 0.5, f'No {geno} Data', ha='center', fontsize=8)
        ax.axis('off')
        return
    
    # Count unique cells for this genotype
    n_cells = len(subset['Cell_ID'].unique())
    
    # Colors
    base_color = 'black' if geno == 'WT' else 'red'
    
    conds = subset['Condition'].unique()
    # Order: Gabazine, then Gabazine + Baclofen
    conds = sorted(conds, key=lambda x: len(x))
    
    for cond in conds:
        cond_data = subset[subset['Condition'] == cond]
        stats = cond_data.groupby('Current_pA')['Firing_Rate_Hz'].agg(['mean', 'sem']).reset_index()
        
        label = 'Gabazine'
        c = base_color
        
        if 'baclofen' in cond.lower():
            label = 'Gab + Bac'
            # Use lighter/different color for post-baclofen
            if geno == 'WT': c = 'gray'
            else: c = 'salmon'
        
        ax.errorbar(stats['Current_pA'], stats['mean'], yerr=stats['sem'], 
                    fmt='o', linestyle='-', color=c, capsize=2, markersize=3, label=label, linewidth=1)
    
    ax.set_xlabel('Current (pA)', fontsize=8)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=8)
    # Add N-number to title
    ax.set_title(f'{title} (n={n_cells})', fontsize=8, fontweight='bold')
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_box_aspect(1)

# ==================================================================================================
# FIGURE 6: THETA BURST / DENDRITIC EXCITABILITY PLOTTING FUNCTIONS
# ==================================================================================================

def plot_theta_raw_traces(fig, gs, raw_data, cols, col_titles, acq_freq=20000, start_idx=10000, end_idx=30000, start_row=0, label="A"):
    """
    Plot raw theta burst traces for Panel A.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        raw_data: dict with structure {genotype: {pathway: trace}}
        cols: list of pathway names
        col_titles: list of column titles
        acq_freq: acquisition frequency in Hz
        start_idx, end_idx: indices for time window
    """
    t_len = end_idx - start_idx
    time = np.arange(t_len) / (acq_freq/1000)
    
    for r, geno in enumerate(['WT', 'I80T/+']):
        row_idx = r
        color = 'k' if geno == 'WT' else 'r'
        # raw_data pkl may use 'GNB1' as key — fall back gracefully
        raw_key = geno if geno in raw_data else ('GNB1' if geno == 'I80T/+' else geno)
        
        for i, pathway in enumerate(cols):
            ax = fig.add_subplot(gs[row_idx, i])
            
            if row_idx == 0:
                ax.set_title(col_titles[i], fontsize=9)
            if row_idx == 0 and i == 0:
                add_subplot_label(ax, label)
            
            # Add row labels on the far left
            if i == 0:
                row_label = "WT" if geno == "WT" else "I80T/+"
                ax.text(-0.3, 0.5, row_label, transform=ax.transAxes, ha='right', va='center', 
                        fontweight='bold', fontsize=9, color='k' if geno == 'WT' else 'r')
            
            if raw_key in raw_data and pathway in raw_data[raw_key]:
                trace = raw_data[raw_key][pathway]
                if len(trace) > len(time):
                    trace = trace[:len(time)]
                
                # BASELINE trace to zero
                trace = trace - np.mean(trace[:min(len(trace), 2000)]) # first 100ms
                
                ax.plot(time[:len(trace)], trace, color=color, linewidth=0.25, label=f'{geno} Raw')
                
                # DRAW dotted line at 20mV relative to baseline (indicated as 20mV on scale)
                ax.axhline(20, color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
            
            ax.axis('off')
            if row_idx == 0 and i == 2:
                add_scale_bar(ax, 200, 10, x_pos=0.7, y_pos=0.1)


def plot_theta_averaged_traces(fig, gs, processed_stats, cols, acq_freq=20000, start_idx=10000, end_idx=30000, start_row=2, label="B"):
    """
    Plot averaged theta traces with SEM and expected traces for Panel B.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        processed_stats: dict with structure {genotype: {pathway: {'mean', 'sem'}, 'Expected': {...}}}
        cols: list of pathway names
        acq_freq: acquisition frequency in Hz
        start_idx, end_idx: indices for time window
    """
    t_len = end_idx - start_idx
    time = np.arange(t_len) / (acq_freq/1000)
    
    # Collect all data to determine shared y-limits
    panel_b_ymin, panel_b_ymax = float('inf'), float('-inf')
    
    for geno in ['WT', 'GNB1']:
        for pathway in cols:
            if pathway in processed_stats.get(geno, {}):
                p_stats = processed_stats[geno][pathway]
                if 'mean' in p_stats:
                    mean_trace = p_stats['mean']
                    sem_trace = p_stats['sem']
                    panel_b_ymin = min(panel_b_ymin, np.nanmin(mean_trace - sem_trace))
                    panel_b_ymax = max(panel_b_ymax, np.nanmax(mean_trace + sem_trace))
                
                if 'Expected_mean' in p_stats:
                    exp_mean = p_stats['Expected_mean']
                    exp_sem = p_stats['Expected_sem']
                    panel_b_ymin = min(panel_b_ymin, np.nanmin(exp_mean - exp_sem))
                    panel_b_ymax = max(panel_b_ymax, np.nanmax(exp_mean + exp_sem))
    
    # Add padding
    y_range = panel_b_ymax - panel_b_ymin
    panel_b_ymin -= 0.1 * y_range
    panel_b_ymax += 0.1 * y_range
    
    for r, geno in enumerate(['WT', 'I80T/+']):
        row_idx = r + 2
        color = 'k' if geno == 'WT' else 'r'
        # processed_stats pkl may use 'GNB1' — fall back gracefully
        stats_key = geno if geno in processed_stats else ('GNB1' if geno == 'I80T/+' else geno)
        
        for i, pathway in enumerate(cols):
            ax = fig.add_subplot(gs[row_idx, i])
            
            if row_idx == start_row and i == 0:
                add_subplot_label(ax, label)
            
            # Add row labels on the far left
            if i == 0:
                row_label = "WT" if geno == "WT" else "I80T/+"
                ax.text(-0.3, 0.5, row_label, transform=ax.transAxes, ha='right', va='center', 
                        fontweight='bold', fontsize=9, color='k' if geno == 'WT' else 'r')
            
            # Plot measured trace with SEM
            if stats_key in processed_stats and pathway in processed_stats[stats_key]:
                p_stats = processed_stats[stats_key][pathway]
                mean_trace = p_stats['mean']
                sem_trace = p_stats['sem']
                
                if len(mean_trace) > len(time):
                    mean_trace = mean_trace[:len(time)]
                    sem_trace = sem_trace[:len(time)]
                
                t = time[:len(mean_trace)]
                
                # Downsample for fill_between
                ds = 10
                t_ds = t[::ds]
                mean_ds = mean_trace[::ds]
                sem_ds = sem_trace[::ds]
                
                ax.fill_between(t_ds, mean_ds - sem_ds, mean_ds + sem_ds, color=color, alpha=0.3, edgecolor='none')
                ax.plot(t_ds, mean_ds, color=color, linewidth=0.25, label=f'{geno}')
                
                # Plot Expected EPSP (Now per pathway)
                if 'Expected_mean' in p_stats:
                    exp_mean = p_stats['Expected_mean']
                    exp_sem = p_stats['Expected_sem']
                    
                    if len(exp_mean) > len(time):
                        exp_mean = exp_mean[:len(time)]
                        exp_sem = exp_sem[:len(time)]
                        
                    t = time[:len(exp_mean)]
                    t_ds = t[::10]
                    exp_ds = exp_mean[::10]
                    exp_sem_ds = exp_sem[::10]
                    
                    # Expected fill between (grey)
                    ax.fill_between(t_ds, exp_ds - exp_sem_ds, exp_ds + exp_sem_ds, color='grey', alpha=0.3, edgecolor='none')
                    ax.plot(t_ds, exp_ds, color='grey', linestyle='-', linewidth=0.25, label='Expected')
                
                # Add threshold line (20mV)
                ax.axhline(20, color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
            
            ax.set_ylim(panel_b_ymin, panel_b_ymax)
            ax.axis('off')
            
            if row_idx == 2 and i == 2:
                add_scale_bar(ax, 200, 10, x_pos=0.7, y_pos=0.1)
            
            if pathway == 'Both' and row_idx == 2: # Only legend on bottom row? Or maybe top?
                # User didn't specify legend location change, but existing was for 'Both'
                ax.legend(frameon=False, fontsize=6, loc='upper right')


def plot_plateau_area_bars_fig6(fig, gs, plateau_df, df_stats=None, start_row=4, label="C", square=True):
    """
    Plot plateau area bar plots for Panel C.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        plateau_df: dataframe with plateau area data
        df_stats: optional stats dataframe for annotations
    """
    valid_conditions = ['Gabazine_Only', 'Before_ML297', 'Before_ETX']
    plateau_filt = plateau_df[plateau_df['Condition'].isin(valid_conditions)]
    
    pathways = ['Perforant', 'Schaffer', 'Both']
    pathway_labels = ['Perforant (ECIII)', 'CA3 (Schaffer)', 'Both Pathways']
    
    # Calculate global range for shared Y-axis
    global_min = plateau_filt['Plateau_Area'].min()
    global_max = plateau_filt['Plateau_Area'].max()
    
    # Increase range slightly for stats
    y_range = global_max - global_min
    y_lim_top = global_max + y_range * 0.3
    y_lim_bottom = min(0, global_min - y_range * 0.1) # Ensure we show at least down to 0, or further if negative
    
    for p_idx, (pathway, label_p) in enumerate(zip(pathways, pathway_labels)):
        ax_bar = fig.add_subplot(gs[start_row, p_idx])
        if p_idx == 0:
            add_subplot_label(ax_bar, label)
        
        pathway_data = plateau_filt[plateau_filt['Pathway'] == pathway].copy()
        
        plot_bar_scatter(ax_bar, pathway_data, 'Genotype', 'Plateau_Area', 'Genotype', order=['WT', 'I80T/+'], unique_col='Cell_ID')
        
        if square:
            ax_bar.set_box_aspect(1)
        
        # Each pathway scales independently — apply_clean_yticks already run inside plot_bar_scatter
        # Ensure bottom is at or below 0 (some plateau areas may be slightly negative)
        apply_clean_yticks(ax_bar)
        
        if p_idx == 0:
            ax_bar.set_ylabel('Plateau Area\n(mV·s)', fontsize=8)
        
        # Add stats annotation
        if df_stats is not None:
            # Map pathway names to match stats file (already matches: Perforant, Schaffer, Both)
            pathway_stats_name = pathway
            match = df_stats[
                (df_stats['Comparison'].str.contains(pathway_stats_name, na=False)) & 
                (df_stats['Figure_Panel'] == 'Fig 6C')
            ]
            
            if not match.empty:
                # Use raw p-value (no FDR - independent hypotheses)
                p_val = match.iloc[0]['P_Value']
                print(f"DEBUG Fig 6C: Pathway={pathway}, Matched={match.iloc[0]['Comparison']}, P={p_val}, Significance={match.iloc[0]['Significance']}")
                y_max = pathway_data['Plateau_Area'].max()
                y_pos = y_max * 1.1
                
                # Only show annotation if p < 0.05
                if p_val < 0.05:
                    ax_bar.plot([0, 0, 1, 1], [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 'k-', linewidth=0.8)
                    
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    ax_bar.text(0.5, y_pos*1.03, sig, ha='center', va='bottom',
                               fontsize=8, fontweight='bold')


def plot_example_difference_traces(fig, gs, supralin_traces, master_df, start_idx=9000, end_idx=30000):
    """
    Plot single WT example with measured, expected, and difference traces for Panel D.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        supralin_traces: dict with supralinearity trace data
        master_df: master dataframe
        start_idx, end_idx: indices for time window
    """
    # Find a WT cell with Both pathway data
    # Use specific WT cell ID if requested
    target_id = '20250325_c2'
    if target_id in supralin_traces:
        wt_example_cell = target_id
    else:
        # Fallback search if target is missing
        for cell_id, cell_data in supralin_traces.items():
            cell_row = master_df[master_df['Cell_ID'] == cell_id]
            if not cell_row.empty and cell_row.iloc[0]['Genotype'] == 'WT':
                if 'Both Pathways' in cell_data:
                    wt_example_cell = cell_id
                    break
    
    if wt_example_cell and 'Both Pathways' in supralin_traces[wt_example_cell]:
        cell_data = supralin_traces[wt_example_cell]['Both Pathways']
        
        # Column 1: Measured + Expected
        ax_d1 = fig.add_subplot(gs[5, 0])
        add_subplot_label(ax_d1, "D")
        
        if 'Measured' in cell_data and 'Expected' in cell_data:
            measured = cell_data['Measured'][start_idx:end_idx]
            expected = cell_data['Expected'][start_idx:end_idx]
            time = np.arange(len(measured)) / 20
            
            ax_d1.plot(time, measured, 'k', linewidth=0.8, label='Actual')
            ax_d1.plot(time, expected, 'gray', linewidth=0.8, label='Expected')
            ax_d1.legend(frameon=False, fontsize=6, loc='upper right')
            # ax_d1.set_title('WT Example', fontsize=8, fontweight='bold') # Removed title
        ax_d1.axis('off')
        
        # Column 2: Difference with annotation
        ax_d2 = fig.add_subplot(gs[5, 1])
        
        if 'Difference' in cell_data:
            supralin = cell_data['Difference'][start_idx:end_idx]
            time = np.arange(len(supralin)) / 20
            
            ax_d2.plot(time, supralin, 'blue', linewidth=0.8)
            # ax_d2.set_title('Actual - Expected = Difference', fontsize=8, fontweight='bold') # Removed title
            
            # Add arrow pointing to peak
            peak_idx = np.argmax(supralin)
            peak_val = supralin[peak_idx]
            peak_time = time[peak_idx]
            ax_d2.annotate('Supralinear', xy=(peak_time, peak_val),
                          xytext=(15, 5), textcoords='offset points',
                          fontsize=7, ha='left', va='bottom',
                          arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))
        ax_d2.axis('off')
        add_scale_bar(ax_d2, 200, 10, x_pos=0.7, y_pos=0.1)
        
        # Column 3: blank
        ax_d3 = fig.add_subplot(gs[5, 2])
        ax_d3.axis('off')
    else:
        ax_d = fig.add_subplot(gs[5, :])
        add_subplot_label(ax_d, "D")
        ax_d.text(0.5, 0.5, "WT example not found", ha='center', va='center')
        ax_d.axis('off')


def plot_averaged_difference_traces(fig, gs, supralin_traces, master_df, start_idx=9000, end_idx=30000, start_row=6, label="E"):
    """
    Plot averaged supralinearity (difference) traces for WT vs GNB1 for Panel E.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        supralin_traces: dict with supralinearity trace data
        master_df: master dataframe
        start_idx, end_idx: indices for time window
    """
    for col_idx, pathway in enumerate(['Perforant', 'Schaffer', 'Both Pathways']):
        ax_e = fig.add_subplot(gs[start_row, col_idx])
        if col_idx == 0:
            add_subplot_label(ax_e, label)
        
        # Collect difference traces per genotype
        wt_diffs = []
        gnb1_diffs = []
        
        for cell_id, cell_data in supralin_traces.items():
            if pathway in cell_data and 'Difference' in cell_data[pathway]:
                cell_row = master_df[master_df['Cell_ID'] == cell_id]
                if not cell_row.empty:
                    geno = cell_row.iloc[0]['Genotype']
                    diff_trace = cell_data[pathway]['Difference']
                    
                    # Slice to window
                    if len(diff_trace) >= end_idx:
                        sliced_trace = diff_trace[start_idx:end_idx]
                    else:
                        sliced_trace = diff_trace[start_idx:]
                    
                    # Add Baseline Correction (100ms = 2000pts)
                    if len(sliced_trace) > 10:
                         sliced_trace = sliced_trace - np.mean(sliced_trace[:min(len(sliced_trace), 2000)])
                    
                    if geno == 'WT':
                        wt_diffs.append(sliced_trace)
                    elif geno in ('GNB1', 'I80T/+'):
                        gnb1_diffs.append(sliced_trace)
        
        # Average and plot WT
        if wt_diffs:
            min_len = min(len(t) for t in wt_diffs)
            wt_arr = np.array([t[:min_len] for t in wt_diffs])
            wt_mean = np.nanmean(wt_arr, axis=0)
            wt_sem = np.nanstd(wt_arr, axis=0) / np.sqrt(len(wt_arr))
            time = np.arange(min_len) / 20
            
            ds = 10
            time_ds = time[::ds]
            wt_mean_ds = wt_mean[::ds]
            wt_sem_ds = wt_sem[::ds]
            
            ax_e.fill_between(time_ds, wt_mean_ds - wt_sem_ds, wt_mean_ds + wt_sem_ds, color='k', alpha=0.3, edgecolor='none')
            ax_e.plot(time_ds, wt_mean_ds, 'k', linewidth=0.25, label=f'WT (n={len(wt_diffs)})')
        
        # Average and plot GNB1
        if gnb1_diffs:
            min_len_g = min(len(t) for t in gnb1_diffs)
            gnb1_arr = np.array([t[:min_len_g] for t in gnb1_diffs])
            gnb1_mean = np.nanmean(gnb1_arr, axis=0)
            gnb1_sem = np.nanstd(gnb1_arr, axis=0) / np.sqrt(len(gnb1_arr))
            time_g = np.arange(min_len_g) / 20
            
            ds = 10
            time_g_ds = time_g[::ds]
            gnb1_mean_ds = gnb1_mean[::ds]
            gnb1_sem_ds = gnb1_sem[::ds]
            
            ax_e.fill_between(time_g_ds, gnb1_mean_ds - gnb1_sem_ds, gnb1_mean_ds + gnb1_sem_ds, color='r', alpha=0.3, edgecolor='none')
            ax_e.plot(time_g_ds, gnb1_mean_ds, 'r', linewidth=0.25, label=f'GNB1 (n={len(gnb1_diffs)})')
        
        # ax_e.set_title(pathway.replace('Both Pathways', 'Both'), fontsize=8, fontweight='bold') # Removed title
        ax_e.axis('off')
        if col_idx == 2:
            ax_e.legend(frameon=False, fontsize=6, loc='upper right')
            add_scale_bar(ax_e, 200, 10, x_pos=0.7, y_pos=0.1)


def plot_supralinear_peak_cycles(fig, gs, df_peaks, df_stats=None, df_anova=None):
    """
    Plot supralinear peak amplitude across theta cycles for Panel F.
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        df_peaks: dataframe with peak amplitude data across cycles
        df_stats: dataframe with stats from Stats_Results_Figure_6.csv
                  Columns: Figure_Panel, Comparison, P_Value, Significance
    """
    pathway_map = {'Perforant': 0, 'Schaffer': 1, 'Both Pathways': 2}
    stats_label_map = {'Perforant': '(ECIII)', 'Schaffer': '(CA3)', 'Both Pathways': '(Both)'}

    # Calculate global min/max including error bars for shared Y-axis
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    cycles = [1, 2, 3, 4, 5]
    cycle_cols = [f'Cycle_{c}' for c in cycles]
    
    # Pre-scan to find data range
    for p in ['Perforant', 'Schaffer', 'Both Pathways']:
        p_data = df_peaks[df_peaks['Pathway'] == p]
        for g in ['WT', 'GNB1', 'I80T/+']:
            g_data = p_data[p_data['Genotype'] == g]
            if g_data.empty: continue
            
            means = g_data[cycle_cols].mean()
            sems = g_data[cycle_cols].std() / np.sqrt(len(g_data))
            
            top = (means + sems).max()
            bottom = (means - sems).min()
            
            if top > global_y_max: global_y_max = top
            if bottom < global_y_min: global_y_min = bottom
    
    # Add padding
    y_range = global_y_max - global_y_min
    if y_range == 0: y_range = 1
    f_ylim_bottom = global_y_min - 0.1 * y_range
    f_ylim_top = global_y_max + 0.15 * y_range  # More room for stars

    for pathway, col_idx in pathway_map.items():
        ax_f = fig.add_subplot(gs[7, col_idx])
        if col_idx == 0:
            add_subplot_label(ax_f, "F")
        
        # Set Shared Limits
        ax_f.set_ylim(f_ylim_bottom, f_ylim_top)
        
        # Add more Y ticks
        apply_clean_yticks(ax_f)
        
        pathway_data = df_peaks[df_peaks['Pathway'] == pathway]
        
        # Store y-max for annotation placement (local to this plot, but bounds are fixed)
        global_max = -np.inf # Used variable name in original code
        
        for geno, color in [('WT', 'k'), ('GNB1', 'r'), ('I80T/+', 'r')]:
            geno_data = pathway_data[pathway_data['Genotype'] == geno]
            
            means = []
            sems = []
            for c in cycles:
                col = f'Cycle_{c}'
                vals = geno_data[col].dropna()
                mean = vals.mean() if len(vals) > 0 else np.nan
                sem = vals.std()/np.sqrt(len(vals)) if len(vals) > 0 else np.nan
                means.append(mean)
                sems.append(sem)
                
                if not np.isnan(mean) and not np.isnan(sem):
                     top = mean + sem
                     if top > global_max: global_max = top
            
            n = len(geno_data)
            ax_f.errorbar(cycles, means, yerr=sems, color=color, marker='o',
                         markersize=3, linewidth=1, capsize=2, label=f'{geno} (n={n})')
            
            # Connect the points with lines explicitly
            ax_f.plot(cycles, means, color=color, linewidth=1)
        
        # ----------------------------------------------------------------------
        # ANNOTATE STATS FROM CSV
        # ----------------------------------------------------------------------
        if df_stats is not None:
            # Filter for Panel F
            panel_stats = df_stats[df_stats['Figure_Panel'] == 'Fig 6F']
            label_tag = stats_label_map[pathway]
            
            # 1. Check Interaction for this pathway
            # String e.g.: "Genotype x Cycle slope ANOVA (CA3)"
            interaction_row = panel_stats[
                panel_stats['Comparison'].str.contains(f"ANOVA {re.escape(label_tag)}")
            ]
            
            is_interaction_sig = False
            if not interaction_row.empty:
                # Use raw P-Value for interaction check (ANOVA usually not FDR corrected against t-tests)
                # But here user corrected everything together. Let's use Significance_FDR column.
                sig_str = interaction_row.iloc[0]['Significance_FDR']
                if sig_str != 'ns':
                    is_interaction_sig = True
            
            # 2. If Interaction is Significant, Plot Post-Hoc Stars
            if is_interaction_sig:
                for c in cycles:
                    # String e.g.: "Cycle 3 (CA3): WT vs GNB1"
                    comp_str = f"Cycle {c} {label_tag}"
                    row = panel_stats[panel_stats['Comparison'].str.contains(re.escape(comp_str))]
                    
                    if not row.empty:
                        sig_symbol = row.iloc[0]['Significance_FDR']
                        if sig_symbol != 'ns':
                            
                            # Calculate local max for this cycle specifically
                            local_max = -np.inf
                            for Geno in ['WT', 'GNB1', 'I80T/+']:
                                g_dat = pathway_data[pathway_data['Genotype'] == Geno][f'Cycle_{c}'].dropna()
                                if not g_dat.empty:
                                    m = g_dat.mean()
                                    s = g_dat.std()/np.sqrt(len(g_dat))
                                    if (m+s) > local_max: local_max = m+s
                            
                            # Place start just above the error bar
                            y_pos = local_max + 0.05 * y_range
                            ax_f.text(c, y_pos, sig_symbol, ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

        ax_f.set_xlabel('Theta Cycle', fontsize=8)
        if col_idx == 0:
            ax_f.set_ylabel('Supralinear AUC\n(mV·s)', fontsize=8)
        # ax_f.set_title(pathway.replace('Both Pathways', 'Both'), fontsize=8, fontweight='bold') # Remove title
        ax_f.spines['top'].set_visible(False)
        ax_f.spines['right'].set_visible(False)
        ax_f.tick_params(labelsize=6)
        ax_f.set_xticks(cycles)
        
        if col_idx == 2:
            ax_f.legend(frameon=False, fontsize=6, loc='upper left')

def plot_single_genotype_gabazine(ax, df_amplitudes, pathway_name, genotype, df_stats=None):
    """
    Plot Gabazine for a single genotype.
    
    Args:
        ax: matplotlib axes
        df_amplitudes: DataFrame with E:I amplitudes
        pathway_name: 'Perforant', 'Schaffer', or 'Basal_Stratum_Oriens'
        genotype: 'WT' or 'GNB1'
        df_stats: DataFrame with ANOVA statistics (optional)
    """
    # ISI values
    isis = [300, 100, 50, 25, 10]
    
    # Collect data for specified genotype
    means = []
    sems = []
    
    for isi in isis:
        data = df_amplitudes[
            (df_amplitudes['Genotype'] == genotype) &
            (df_amplitudes['Pathway'] == pathway_name) &
            (df_amplitudes['ISI'] == isi)
        ]['Gabazine_Amplitude'].dropna()
        means.append(data.mean() if len(data) > 0 else np.nan)
        sems.append(data.sem() if len(data) > 0 else 0)
    
    # Plot line (WT=black, GNB1=red)
    color = 'black' if genotype == 'WT' else 'red'
    ax.errorbar(range(len(isis)), means, yerr=sems, color=color, marker='o',
                markersize=3, linewidth=1, label=genotype, capsize=2, capthick=0.5)
    
    # Format axes
    ax.set_xticks(range(len(isis)))
    ax.set_xticklabels([str(isi) for isi in isis])
    ax.set_xlabel('ISI (ms)', fontsize=8)
    ax.set_ylabel('EPSP Amplitude (mV)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='best', fontsize=7)
def plot_supralinear_auc_bars_fig6(fig, gs, auc_total_df, df_stats=None, start_row=7, label="F", square=True):
    """
    Plot supralinear total AUC bar plots for Panel F (like Panel C).
    
    Parameters:
        fig: matplotlib figure
        gs: gridspec
        auc_total_df: dataframe with Total_AUC column
        df_stats: optional stats dataframe for annotations
    """
    pathways = ['Perforant', 'Schaffer', 'Both Pathways']
    pathway_labels = ['Perforant (ECIII)', 'CA3 (Schaffer)', 'Both Pathways']
    
    # Calculate global min and max for shared Y-axis (allow negative values for sublinearity)
    global_min = auc_total_df['Total_AUC'].min()
    global_max = auc_total_df['Total_AUC'].max()
    
    # Add headroom for stats annotations
    y_range = global_max - global_min
    y_lim_bottom = global_min - (y_range * 0.1) if global_min < 0 else 0
    y_lim_top = global_max + (y_range * 0.3)  # Extra room at top for significance bars
    
    for p_idx, (pathway, label_p) in enumerate(zip(pathways, pathway_labels)):
        ax_bar = fig.add_subplot(gs[start_row, p_idx])
        if p_idx == 0:
            add_subplot_label(ax_bar, label)
        
        pathway_data = auc_total_df[auc_total_df['Pathway'] == pathway].copy()
        
        # --- Extract N numbers from stats to ensure Figure matches Text ---
        n_override = None
        if df_stats is not None:
             pathway_stats_name = pathway.replace('Perforant', 'ECIII').replace('Schaffer', 'CA3').replace('Both Pathways', 'Both')
             match = df_stats[
                (df_stats['Comparison'].str.contains(pathway_stats_name, na=False)) & 
                (df_stats['Figure_Panel'] == 'Fig 6F')
             ]
             if not match.empty:
                 row = match.iloc[0]
                 if 'N_WT' in row.index and 'N_GNB1' in row.index:
                     n_override = {'WT': int(row['N_WT']), 'I80T/+': int(row['N_GNB1'])}

        plot_bar_scatter(ax_bar, pathway_data, 'Genotype', 'Total_AUC', 'Genotype', order=['WT', 'I80T/+'], unique_col='Cell_ID', override_n_counts=n_override)
        
        if square:
            ax_bar.set_box_aspect(1)
        
        # ax_bar.set_title(label, fontsize=8, fontweight='bold') # Removed title
        ax_bar.set_ylim(y_lim_bottom, y_lim_top)
        
        apply_clean_yticks(ax_bar)
        
        if p_idx == 0:
            ax_bar.set_ylabel('Supralinear AUC\\n(mV·s)', fontsize=8)
        
        # Add stats annotation
        if df_stats is not None:
            # Map pathway names to match stats file
            pathway_stats_name = pathway.replace('Perforant', 'ECIII').replace('Schaffer', 'CA3').replace('Both Pathways', 'Both')
            
            # Filter specifically for Panel F stats to avoid confusion with Panel C
            match = df_stats[
                (df_stats['Comparison'].str.contains(pathway_stats_name, na=False)) & 
                (df_stats['Figure_Panel'] == 'Fig 6F')
            ]
            
            if not match.empty:
                # Use raw p-value (no FDR - independent hypotheses)
                p_val = match.iloc[0]['P_Value']
                print(f"DEBUG Fig 6F: Pathway={pathway}, Matched={match.iloc[0]['Comparison']}, P={p_val}, Significance={match.iloc[0]['Significance']}")
                
                # Calculate y_pos based on actual plot limits (handles negative values)
                y_max_plot = y_lim_top  # Use the ylim_top calculated globally
                y_pos = y_max_plot * 0.85
                
                # Only show annotation if p < 0.05
                if p_val < 0.05:
                    ax_bar.plot([0, 0, 1, 1], [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 'k-', linewidth=0.8)
                    
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    ax_bar.text(0.5, y_pos*1.03, sig, ha='center', va='bottom',
                               fontsize=8, fontweight='bold')

# -------------------------------------------------------------------------
# DATA PREPARATION HELPERS
# -------------------------------------------------------------------------

def prepare_figure_7_data(df_auc_total=None):
    """
    Loads and processes data for Figure 7 (Panels A, B, and C/F filtering).
    Moves calculation logic out of generate_figures.py.
    
    Returns:
        raw_data (Panel A traces)
        processed_stats (Panel B mean/sem)
        plateau_df (Panel C dataframe)
        df_auc_total (Filtered Panel F dataframe)
    """
    import os
    import pickle
    import pandas as pd
    import numpy as np
    
    # 1. Configuration
    # -------------------------------------------------------------------------
    acq_freq = 20000
    # Supralinear traces are pre-cropped to start at 400ms
    # So start_idx=0 (beginning of cropped trace), end_idx = (1500-400)*20 = 22000
    start_ms = 400  # For Panel A (raw traces, NOT pre-cropped)
    end_ms = 1500
    start_idx = 0  # Traces already start at 400ms
    end_idx = int((end_ms - start_ms) * acq_freq / 1000)  # 22000 samples
    # Keep original indices for Panel A raw traces (which are NOT pre-cropped)
    raw_start_idx = int(start_ms * acq_freq / 1000)  # 8000
    raw_end_idx = int(end_ms * acq_freq / 1000)  # 30000
    
    # 2. Load Data for Panel A (Raw Traces)
    # -------------------------------------------------------------------------
    example_traces_path = os.path.join('paper_data', 'Plateau_data', 'Figure7_Example_Traces.pkl')
    raw_data = {'WT': {}, 'I80T/+': {}}
    
    if os.path.exists(example_traces_path):
        with open(example_traces_path, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"  Loaded example traces from {example_traces_path}")
    else:
        print(f"  ⚠ Warning: Example traces not found at {example_traces_path}")

    # 3. Load Data for Panel B (Averaged + Expected Traces)
    # -------------------------------------------------------------------------
    supralin_trace_path = os.path.join('paper_data', 'supralinearity', 'Supralinear_Traces_Plotting.pkl')
    supralin_traces = {}
    
    avg_data = {'WT': {'Perforant': [], 'Schaffer': [], 'Both': []}, 
                'GNB1': {'Perforant': [], 'Schaffer': [], 'Both': []}}
    # Store Expected separately per pathway
    expected_data = {'WT': {'Perforant': [], 'Schaffer': [], 'Both': []}, 
                     'GNB1': {'Perforant': [], 'Schaffer': [], 'Both': []}}
    
    if os.path.exists(supralin_trace_path):
        supralin_traces = pd.read_pickle(supralin_trace_path)
        
        # Load Master DF locally
        if os.path.exists('master_df.csv'):
            master_df_temp = pd.read_csv('master_df.csv', low_memory=False)
        else:
            print("  Warning: master_df.csv not found for trace aggregation.")
            master_df_temp = pd.DataFrame()
            
        if not master_df_temp.empty:
            for cell_id, cell_data in supralin_traces.items():
                cell_row = master_df_temp[master_df_temp['Cell_ID'] == cell_id]
                if cell_row.empty: continue
                geno = cell_row.iloc[0]['Genotype']
                if geno not in ['WT', 'GNB1']: continue
                
                for pathway in ['Perforant', 'Schaffer', 'Both Pathways']:
                    short_path = pathway.replace(' Both Pathways', 'Both').replace('Both Pathways', 'Both')
                    if pathway in cell_data:
                        # Collect Measured
                        if 'Measured' in cell_data[pathway]:
                            meas = cell_data[pathway]['Measured']
                            if len(meas) >= end_idx:
                                meas_seg = meas[start_idx:end_idx]
                                # Add Baseline Correction (100ms = 2000pts)
                                meas_seg = meas_seg - np.mean(meas_seg[:2000])
                                avg_data[geno][short_path].append(meas_seg)
                        
                        # Collect Expected (For ALL pathways, not just Both)
                        if 'Expected' in cell_data[pathway]:
                            exp = cell_data[pathway]['Expected']
                            if len(exp) >= end_idx:
                                exp_seg = exp[start_idx:end_idx]
                                # Add Baseline Correction (100ms = 2000pts)
                                exp_seg = exp_seg - np.mean(exp_seg[:2000])
                                expected_data[geno][short_path].append(exp_seg)

        # print(f"  Panel B (Checked): WT cells={len(avg_data['WT']['Both'])}, GNB1 cells={len(avg_data['GNB1']['Both'])}")
    
    # Calculate averages and SEM
    processed_stats = {'WT': {}, 'GNB1': {}}
    for geno in ['WT', 'GNB1']:
        for pathway in ['Perforant', 'Schaffer', 'Both']:
            processed_stats[geno][pathway] = {}
            
            # Measured Stats
            if len(avg_data[geno][pathway]) > 0:
                min_len = min(len(t) for t in avg_data[geno][pathway])
                arr = np.array([t[:min_len] for t in avg_data[geno][pathway]])
                processed_stats[geno][pathway]['mean'] = np.mean(arr, axis=0)
                processed_stats[geno][pathway]['sem'] = np.std(arr, axis=0) / np.sqrt(len(arr))
            
            # Expected Stats (Per Pathway)
            if len(expected_data[geno][pathway]) > 0:
                min_len = min(len(t) for t in expected_data[geno][pathway])
                arr = np.array([t[:min_len] for t in expected_data[geno][pathway]])
                
                processed_stats[geno][pathway]['Expected_mean'] = np.mean(arr, axis=0)
                processed_stats[geno][pathway]['Expected_sem'] = np.std(arr, axis=0) / np.sqrt(len(arr))

    # Load Plateau Data for Panel E (Plateau Area — uses 20mV threshold)
    plateau_csv_path = os.path.join('paper_data', 'Plateau_data', 'Plateau_data.csv')
    plateau_df = pd.DataFrame()
    if os.path.exists(plateau_csv_path):
        plateau_df = pd.read_csv(plateau_csv_path)
        plateau_df['Cell_ID'] = plateau_df['Cell_ID'].astype(str)

    # Panel D (Supralinear AUC) uses cells from Supralinear_AUC_Total.csv,
    # filtered by master_df inclusion criteria:
    #   1. 'Inclusion' column must contain 'plateau' (all pathways)
    #   2. 'Single Pathway Plateau Inclusion' == 'Yes' (Schaffer/Perforant — enforced at export time)
    # Panel D is INDEPENDENT of Panel E — no 20mV plateau threshold cross-filter.
    if df_auc_total is not None and not df_auc_total.empty:
        df_auc_total['Cell_ID'] = df_auc_total['Cell_ID'].astype(str)
        # Safety filter: only keep cells whose master_df Inclusion contains 'plateau'
        if os.path.exists('master_df.csv'):
            mdf = pd.read_csv('master_df.csv', low_memory=False)
            plateau_ok = set(
                mdf[mdf['Inclusion'].astype(str).str.contains('plateau', case=False, na=False)]['Cell_ID'].astype(str)
            )
            before = len(df_auc_total)
            df_auc_total = df_auc_total[df_auc_total['Cell_ID'].isin(plateau_ok)].copy()
            print(f"  Panel D: {len(df_auc_total)} rows for Supralinear AUC (filtered {before - len(df_auc_total)} non-plateau cells).")
        else:
            print(f"  Panel D: Using {len(df_auc_total)} rows for Supralinear AUC (master_df not found for filtering).")


    return raw_data, processed_stats, plateau_df, df_auc_total, supralin_traces


def plot_girk_delta_bars(ax, df_delta, drug_name, stats_df=None):
    """
    Plots Delta Change grouped by Pathway (Both, Schaffer, Perforant).
    """
    pathways = ['Both']
    genotypes = ['WT', 'I80T/+']
    
    x_start = 0
    group_spacing = 2.5
    bar_width = 0.6
    bar_gap = 0.05
    
    ticks = []
    tick_labels = []
    
    for i, path in enumerate(pathways):
        # Filter data for this pathway
        sub = df_delta[(df_delta['Drug'] == drug_name) & (df_delta['Pathway'] == path)].copy()
        
        # Center of this group
        # WT at x, GNB1 at x + bar_width + bar_gap
        wt_x = x_start
        gnb1_x = x_start + bar_width + bar_gap
        
        group_center = (wt_x + gnb1_x) / 2
        ticks.append(group_center)
        tick_labels.append(path)
        
        # Plot WT
        wt_data = sub[sub['Genotype'] == 'WT']['Delta_Area']
        if len(wt_data) > 0:
            ax.bar(wt_x, wt_data.mean(), width=bar_width, color='black', alpha=0.5)
            ax.scatter([wt_x]*len(wt_data), wt_data, color='black', s=5, zorder=3)
            ax.errorbar(wt_x, wt_data.mean(), yerr=wt_data.sem(), fmt='none', color='black', capsize=2)
            
        # Plot I80T/+ (data may be stored as 'GNB1' or 'I80T/+')
        gnb1_data = sub[sub['Genotype'].isin(['GNB1', 'I80T/+'])]['Delta_Area']
        if len(gnb1_data) > 0:
            ax.bar(gnb1_x, gnb1_data.mean(), width=bar_width, color='red', alpha=0.5)
            ax.scatter([gnb1_x]*len(gnb1_data), gnb1_data, color='red', s=5, zorder=3)
            ax.errorbar(gnb1_x, gnb1_data.mean(), yerr=gnb1_data.sem(), fmt='none', color='red', capsize=2)

        # Stats
        if stats_df is not None:
             row = stats_df[(stats_df['Drug'] == drug_name) & (stats_df['Pathway'] == path)]
             if not row.empty:
                 p_val = row.iloc[0]['p_value']
                 # Determine y_max to place stats
                 y_vals = sub['Delta_Area']
                 if not y_vals.empty:
                     ymax = y_vals.max()
                     if np.isnan(ymax): ymax = 0
                     
                     # Calculate bracket height
                     # Ensure bracket is above data
                     # If ymax is negative, we still want bracket above the highest point
                     # Add a buffer
                     bracket_y = ymax + (ymax * 0.1 if ymax > 0 else 1.0)
                     if ymax < 0 and bracket_y < 0: bracket_y = 1.0 # Force positive if all negative? No, bracket can be at -1.
                     # But usually stats bracket is at top of bars.
                     # For simplicity, if everything is negative, put at 0 + offset?
                     # Let's use max(ymax, 0) + constant
                     peak = max(ymax, 0) if len(y_vals) > 0 else 0
                     bracket_y = peak + 2 # Add 2 mV*ms buffer
                     
                     draw_significance(ax, wt_x, gnb1_x, p_val, bracket_y)

        x_start += group_spacing
    
    # Label each bar directly on the x-axis: 'WT' and 'I80T/+' under respective bars
    bar_xticks = []
    bar_xlabels = []
    x_pos = 0
    for path in pathways:
        bar_xticks.extend([x_pos, x_pos + bar_width + bar_gap])
        bar_xlabels.extend(['WT', 'I80T/+'])
        x_pos += group_spacing
    ax.set_xticks(bar_xticks)
    ax.set_xticklabels(bar_xlabels, fontsize=7)
    ax.set_ylabel('Delta Area (mV*ms)', fontsize=8)
    ax.set_title(drug_name, fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

def plot_traces_GIRK_v2(ax, traces_before, traces_after, genotype, drug_name, 
                        after_color='gold', add_legend=False, add_scale=False):
    """
    Plot Before (Black) vs After (Color) traces.
    Replaces plot_traces_GIRK_exp with specific color logic.
    """
    color_before = 'black'
    color_after = after_color
    acq_freq = 20000
    
    before_list = []
    after_list = []
    
    # Extraction Logic — pkl stores genotype as 'GNB1', display uses 'I80T/+'
    # Fall back to 'GNB1' when looking up data if display label is 'I80T/+'
    data_geno = genotype if genotype != 'I80T/+' else 'GNB1'

    for cell_id, cell_data in traces_before.items():
        if cell_data['genotype'] in (genotype, data_geno):
            if 'Both' in cell_data['traces']:
                trace = cell_data['traces']['Both']
                if len(trace) >= 30000:
                    before_list.append(trace[10000:30000])
    
    for cell_id, cell_data in traces_after.items():
        if cell_data['genotype'] in (genotype, data_geno):
            if 'Both' in cell_data['traces']:
                trace = cell_data['traces']['Both']
                if len(trace) >= 30000:
                    after_list.append(trace[10000:30000])
    
    if before_list and after_list:
        before_mean = np.mean(before_list, axis=0)
        before_sem = np.std(before_list, axis=0) / np.sqrt(len(before_list))
        after_mean = np.mean(after_list, axis=0)
        after_sem = np.std(after_list, axis=0) / np.sqrt(len(after_list))
        
        # Downsample for Plotting
        ds = 20
        before_mean = before_mean[::ds]
        before_sem = before_sem[::ds]
        after_mean = after_mean[::ds]
        after_sem = after_sem[::ds]
        time = np.arange(len(before_mean)) * ds / (acq_freq / 1000)
        
        # Plot Before (Black)
        ax.fill_between(time, before_mean - before_sem, before_mean + before_sem, 
                        color=color_before, alpha=0.3, edgecolor='none')
        ax.plot(time, before_mean, color=color_before, linewidth=1.2, label='Before')
        
        # Plot After (Custom Color)
        ax.fill_between(time, after_mean - after_sem, after_mean + after_sem, 
                        color=color_after, alpha=0.3, edgecolor='none')
        ax.plot(time, after_mean, color=color_after, linewidth=1.2, label=f'After {drug_name}')
        
        if add_scale:
            add_scale_bar(ax, 200, 10, x_pos=0.7, y_pos=0.1)
            
        ax.set_title(genotype, fontweight='bold', fontsize=9)
        if add_legend:
             ax.legend(frameon=False, fontsize=7, loc='upper right')
             
        ax.axhline(20, color='gray', linestyle=':', linewidth=1.0, zorder=0)
        ax.text(time[0], 20, '20 mV', va='center', ha='right', fontsize=6, color='gray')
             
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No traces', ha='center', axis='off')
        ax.axis('off')
    

def plot_PPR_by_genotype_and_channel(df, output_path):
    """
    Plot PPR data by genotype and channel using plot_bar_scatter style.
    Creates one subplot per channel, each with WT vs I80T/+ bar+scatter.
    
    Parameters:
        df: DataFrame with columns: Cell_ID, Genotype, Channel_Label, PPR
        output_path: Path to save the plot (.png). Will also save .svg.
    """
    import scipy.stats as sci_stats
    
    print("\nPlotting PPR data...")
    
    plot_df = df.dropna(subset=['PPR']).copy()
    plot_df = rename_genotype(plot_df, 'Genotype')
    
    if plot_df.empty:
        print("⚠ No valid PPR data to plot.")
        return
        
    channels = sorted(plot_df['Channel_Label'].unique())
    genotype_order = ['WT', 'I80T/+']
    
    fig, axes = plt.subplots(1, len(channels), figsize=(3.5 * len(channels), 4))
    if len(channels) == 1:
        axes = [axes]
    
    for ax, ch_label in zip(axes, channels):
        ch_df = plot_df[plot_df['Channel_Label'] == ch_label]
        
        max_h = plot_bar_scatter(ax, ch_df, x_col='Genotype', y_col='PPR', 
                                  hue_col='Genotype', order=genotype_order,
                                  unique_col='Cell_ID')
        
        ax.set_title(ch_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Paired Pulse Ratio')
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Stats: Mann-Whitney U
        wt_vals = ch_df[ch_df['Genotype'] == 'WT']['PPR'].dropna()
        gnb1_vals = ch_df[ch_df['Genotype'] == 'I80T/+']['PPR'].dropna()
        
        if len(wt_vals) > 2 and len(gnb1_vals) > 2:
            stat, p_val = sci_stats.mannwhitneyu(wt_vals, gnb1_vals, alternative='two-sided')
            sig = '****' if p_val < 0.0001 else '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            y_bar = max_h * 1.15
            ax.plot([0, 0, 1, 1], [y_bar, y_bar * 1.02, y_bar * 1.02, y_bar], lw=1.2, c='k')
            ax.text(0.5, y_bar * 1.03, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_ylim(top=y_bar * 1.2)
            print(f"  {ch_label} WT vs I80T/+: p={p_val:.4f} ({sig})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    with plt.rc_context({'svg.fonttype': 'none'}):
        plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"✓ PPR plot saved to: {output_path} (.png and .svg)")


def plot_PPR_examples(data_dir, df, output_dir):
    """
    Plot 4 example PPR traces. Uses offset_trace (artifact-removed) when available,
    otherwise falls back to raw sweep. Shows detected stim locations.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from analysis_utils import convert_filename_to_standard_id
    
    os.makedirs(output_dir, exist_ok=True)
    
    plotted = 0
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    valid_ids = set(df['Cell_ID'].astype(str))
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.pkl'): continue
        cell_id = convert_filename_to_standard_id(filename)
        if cell_id is None or cell_id not in valid_ids: continue
        
        filepath = os.path.join(data_dir, filename)
        file_df = pd.read_pickle(filepath)
        
        for _, row in file_df.iterrows():
            sweep = row.get('sweep')
            if sweep is None or not isinstance(sweep, np.ndarray): continue
            
            acq_freq = row.get('acquisition_frequency', 20000.0)
            diff_sw = np.abs(np.diff(sweep))
            peaks, _ = find_peaks(diff_sw, height=5.0, distance=int(acq_freq * 0.01))
            
            if len(peaks) == 0: continue
            times_ms = peaks / acq_freq * 1000
            valid_times = sorted([t for t in times_ms if t > 250])
            if len(valid_times) not in [2, 4]: continue
            
            diffs_arr = np.diff(valid_times)
            has_50 = any(abs(d - 50) < 5 for d in diffs_arr)
            is_theta = (any(abs(d - 10) < 5 for d in diffs_arr) or 
                       any(abs(d - 20) < 5 for d in diffs_arr))
            if not has_50 or is_theta: continue
            
            ax = axes[plotted]
            
            # Try offset_trace first
            int_traces = row.get('intermediate_traces')
            used_offset = False
            if type(int_traces) is dict:
                offset = int_traces.get('offset_trace')
                if type(offset) is dict:
                    for ch, ch_data in offset.items():
                        if isinstance(ch_data, np.ndarray) and len(ch_data) > 100:
                            time_axis = np.arange(len(ch_data)) / acq_freq * 1000
                            ax.plot(time_axis, ch_data, color='black', lw=1.0)
                            ax.axvline(50, color='blue', ls='--', alpha=0.5, lw=0.8, label='50ms ISI')
                            ax.set_title(f"{cell_id} | {ch} (offset_trace)", fontsize=9)
                            ax.set_ylabel('Amplitude (mV)')
                            ax.set_xlabel('Time from stim (ms)')
                            used_offset = True
                            break
            
            if not used_offset:
                time_axis = np.arange(len(sweep)) / acq_freq * 1000
                ax.plot(time_axis, sweep, color='black', lw=0.8)
                for t in valid_times:
                    ax.axvline(t, color='red', ls='--', alpha=0.5, lw=0.8)
                ax.set_title(f"{cell_id} | raw sweep", fontsize=9)
                ax.set_ylabel('Voltage (mV)')
                ax.set_xlabel('Time (ms)')
                ax.set_xlim(min(valid_times) - 50, max(valid_times) + 150)
                start_i = int(max(0, (min(valid_times) - 50) * acq_freq / 1000))
                end_i = int(min(len(sweep), (max(valid_times) + 150) * acq_freq / 1000))
                y_min, y_max = np.min(sweep[start_i:end_i]), np.max(sweep[start_i:end_i])
                ax.set_ylim(y_min - 2, y_max + 2)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plotted += 1
            break
        
        if plotted >= 4:
            break
            
    if plotted > 0:
        plt.tight_layout()
        example_path = os.path.join(output_dir, 'PPR_AutoExtract_Examples.png')
        plt.savefig(example_path, dpi=300)
        plt.close()
        print(f"✓ Plotted {plotted} example traces to {example_path}")


# ==================================================================================================
# FIGURE 7: GIRK PHARMACOLOGY & UNITARY GABAB
# ==================================================================================================

def plot_unitary_gabab_traces_by_pathway(ax, traces_pkl_path, genotype, pathway, label, drugs=None):
    """
    Plot overlaid unitary GABAB traces for one genotype + one pathway.
    Colors are consistent with the rest of Figure 7:
      - Gabazine baseline: black (WT) / dark red (GNB1)
      - ML297: gold  (matches Panel F after_color)
      - ETX:   cyan  (matches Panel H after_color)
    Axes are hidden; a scale bar is added instead.
    """
    add_subplot_label(ax, label)
    if not os.path.exists(traces_pkl_path):
        ax.text(0.5, 0.5, 'No Traces Data', ha='center', va='center', transform=ax.transAxes)
        return

    with open(traces_pkl_path, 'rb') as f:
        traces_dict = pickle.load(f)

    gab_traces, ml297_traces, etx_traces = [], [], []

    for cid, info in traces_dict.items():
        if info['Genotype'] != genotype:
            continue
        for cond, paths in info['Traces'].items():
            if not isinstance(paths, dict): continue
            trace = paths.get(pathway)
            if trace is None: continue
            c_lower = cond.lower()
            if 'ml297' in c_lower or 'ml-297' in c_lower:
                ml297_traces.append(trace)
            elif 'etx' in c_lower:
                etx_traces.append(trace)
            elif 'gabazine' in c_lower or 'control' in c_lower:
                gab_traces.append(trace)

    if not any([gab_traces, ml297_traces, etx_traces]):
        ax.text(0.5, 0.5, f'No {genotype}/{pathway} traces',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    dt = 1 / 20  # ms (20 kHz sampling)
    is_wt = (genotype == 'WT')

    # Gabazine: genotype color. ML297: gold. ETX: cyan. Consistent with F & H.
    gab_color = 'black' if is_wt else '#8B0000'
    
    configs = []
    if drugs is None or 'Gabazine' in drugs:
        configs.append((gab_traces, gab_color, f'{genotype} Gabazine'))
    if drugs is None or 'ML297' in drugs:
        configs.append((ml297_traces, 'goldenrod', f'{genotype} ML297'))
    if drugs is None or 'ETX' in drugs:
        configs.append((etx_traces, 'darkcyan', f'{genotype} ETX'))

    all_values = []
    for traces, color, lbl in configs:
        if not traces: continue
        min_len = min(len(t) for t in traces)
        arr = np.array([np.array(t[:min_len]) for t in traces])
        mean = np.mean(arr, axis=0)
        sem  = np.std(arr, axis=0) / np.sqrt(len(arr))
        time = np.arange(len(mean)) * dt
        ax.plot(time, mean, color=color, label=f'{lbl} (n={len(traces)})', lw=1.2)
        ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.15)
        all_values.extend(mean.tolist())

    ax.set_xlim(-10, 350)
    ax.axhline(0, color='lightblue', lw=0.5, zorder=0)
    pathway_label = 'Perforant Path' if pathway == 'Perforant' else 'Schaffer Collateral'
    ax.set_title(f'{genotype} – {pathway_label}', fontsize=8, fontweight='bold')
    ax.legend(frameon=False, fontsize=5.5, loc='upper right')

    # Remove all axes
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust ylim to make trace larger (less expansion than 1.5)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        overall_range = y_max - y_min
        center = (y_max + y_min) / 2
        # Use 1.2 instead of 1.5 to make it larger
        ax.set_ylim(center - overall_range * 1.2 - 0.5, center + overall_range * 1.2 + 0.5)
        
        sb_x = 250
        sb_y = center - overall_range * 1.0 - 0.2
        ax.plot([sb_x, sb_x + 50], [sb_y, sb_y], 'k-', lw=1.5, clip_on=False)
        ax.plot([sb_x + 50, sb_x + 50], [sb_y, sb_y + 2], 'k-', lw=1.5, clip_on=False)
        ax.text(sb_x + 25, sb_y - overall_range * 0.1, '50 ms', ha='center', va='top', fontsize=5.5)
        ax.text(sb_x + 55, sb_y + 1, '2 mV', ha='left', va='center', fontsize=5.5)


def plot_unitary_gabab_traces_combined(ax, traces_pkl_path, drug_name, pathway_name, label):
    """
    Plot overlaid unitary GABAB traces for BOTH genotypes for one pathway.
    """
    add_subplot_label(ax, label)
    if not os.path.exists(traces_pkl_path):
        ax.text(0.5, 0.5, 'No Traces Data', ha='center', va='center', transform=ax.transAxes)
        return

    import pickle
    import numpy as np
    import pandas as pd
    with open(traces_pkl_path, 'rb') as f:
        traces_dict = pickle.load(f)

    wt_gab, wt_drug, gnb_gab, gnb_drug = [], [], [], []

    for cid, info in traces_dict.items():
        geno = info['Genotype']
        for cond, paths in info['Traces'].items():
            if not isinstance(paths, dict): continue
            trace = paths.get(pathway_name)
            if trace is None: continue
            c_lower = cond.lower()
            
            if drug_name.lower() in c_lower:
                if geno == 'WT': wt_drug.append(trace)
                else: gnb_drug.append(trace)
            elif 'gabazine' in c_lower or 'control' in c_lower:
                if geno == 'WT': wt_gab.append(trace)
                else: gnb_gab.append(trace)

    dt = 1 / 20  # ms
    configs = [
        (wt_gab, 'black', 'WT Gabazine', '-'),
        (wt_drug, 'gray', f'WT {drug_name}', '--'), 
        (gnb_gab, '#8B0000', 'I80T/+ Gabazine', '-'),
        (gnb_drug, 'salmon', f'I80T/+ {drug_name}', '--'),
    ]

    all_values = []
    for traces, color, lbl, ls in configs:
        if not traces: continue
        min_len = min(len(t) for t in traces)
        arr = np.array([np.array(t[:min_len]) for t in traces])
        mean = np.mean(arr, axis=0)
        sem  = np.std(arr, axis=0) / np.sqrt(len(arr))
        time = np.arange(len(mean)) * dt
        ax.plot(time, mean, color=color, label=f'{lbl}', lw=1.2, linestyle=ls)
        # Use fill_between with a lighter alpha for combined plot visibility
        ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.1)
        all_values.extend(mean.tolist())

    ax.set_xlim(0, 310)
    ax.axhline(0, color='lightblue', lw=0.5, zorder=0)
    path_label = 'Perforant Path' if pathway_name == 'Perforant' else 'Schaffer Collateral'
    ax.set_title(f'WT vs I80T/+ – {path_label}', fontsize=9, fontweight='bold')
    ax.legend(frameon=False, fontsize=5.5, loc='upper right')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if all_values:
        y_min = min(all_values)
        y_range = max(all_values) - y_min
        sb_x = 250
        sb_y = y_min - y_range * 0.05
        ax.plot([sb_x, sb_x + 50], [sb_y, sb_y], 'k-', lw=1.5, clip_on=False)
        ax.plot([sb_x + 50, sb_x + 50], [sb_y, sb_y + 2], 'k-', lw=1.5, clip_on=False)
        ax.text(sb_x + 25, sb_y - y_range * 0.08, '50 ms', ha='center', va='top', fontsize=5.5)
        ax.text(sb_x + 55, sb_y + 1, '2 mV', ha='left', va='center', fontsize=5.5)



def plot_unitary_gabab_area_paired_combined(ax, delta_csv_path, drug_name, pathway_name, label, df_stats=None):
    """
    Plot paired Pre and Post GABAB Area for both WT and I80T/+ on the same plot.
    """
    import scipy.stats as sci_stats
    add_subplot_label(ax, label)
    if not os.path.exists(delta_csv_path):
        ax.text(0.5, 0.5, 'No Delta Data', ha='center', va='center', transform=ax.transAxes)
        return

    df = pd.read_csv(delta_csv_path)
    sub = df[(df['Drug'] == drug_name) & (df['Pathway'] == pathway_name)].copy()

    if sub.empty:
        ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    sub_wt = sub[sub['Genotype'] == 'WT']
    sub_gnb1 = sub[sub['Genotype'].isin(['GNB1', 'I80T/+'])]

    cat1, cat2 = 'Pre_GABAB_Area', 'Post_GABAB_Area'
    
    def plot_paired(data, color, base_x):
        x1, x2 = base_x, base_x + 1
        for _, row in data.iterrows():
            if pd.notna(row[cat1]) and pd.notna(row[cat2]):
                # The areas are negative, plot them as positive (multiply by -1)
                y1 = -row[cat1]
                y2 = -row[cat2]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.3, linewidth=0.8)

        if len(data) > 0:
            m1, m2 = -data[cat1].mean(), -data[cat2].mean()
            s1, s2 = data[cat1].sem(), data[cat2].sem()
            ax.plot([x1, x2], [m1, m2], color=color, linewidth=2, alpha=1.0)
            ax.errorbar(x1, m1, yerr=s1, fmt='o', color=color, capsize=2, markersize=4)
            ax.errorbar(x2, m2, yerr=s2, fmt='o', color=color, capsize=2, markersize=4)

    plot_paired(sub_wt, 'black', 0)
    plot_paired(sub_gnb1, 'red', 2.5)

    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(['Gabazine', f'{drug_name}', 'Gabazine', f'{drug_name}'], fontsize=7, rotation=0)

    # Add WT and I80T/+ labels under the groups
    ax.text(0.5, -0.15, 'WT (Black)', ha='center', va='top', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=8)
    ax.text(3.0, -0.15, 'I80T/+ (Red)', ha='center', va='top', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=8)

    ax.set_xlim(-0.5, 4.0)
    ax.set_ylabel('Slow IPSP Area (mV·s)', fontsize=8)
    ax.set_title(f'WT vs I80T/+ {pathway_name}', fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Statistical annotation (across genotype)
    if df_stats is not None:
        comparison = f"WT vs GNB1 ({pathway_name} Unitary Delta)"
        row = df_stats[(df_stats['Drug'] == drug_name) & (df_stats['Comparison'] == comparison)]
        if not row.empty:
            sig = row.iloc[0].get('Significance', 'ns')
            
            y_max = 0
            if len(sub_wt) > 0:
                y_max = max(y_max, max((-sub_wt[cat1]).max(), (-sub_wt[cat2]).max()))
            if len(sub_gnb1) > 0:
                y_max = max(y_max, max((-sub_gnb1[cat1]).max(), (-sub_gnb1[cat2]).max()))
                
            y_pos = y_max * 1.05 + 0.05
            ax.plot([0.5, 3.0], [y_pos, y_pos], 'k-', lw=0.8)
            ax.text(1.75, y_pos*1.02, sig, ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(bottom=0, top=y_pos*1.2)


def plot_unitary_gabab_area_delta_single(ax, delta_csv_path, drug_name, pathway_name, label, df_stats=None):
    """
    Plot Delta GABAB Area for a single drug + pathway combination.
    Used to create 4 separate subplots in Panel C.
    """
    add_subplot_label(ax, label)
    if not os.path.exists(delta_csv_path):
        ax.text(0.5, 0.5, 'No Delta Data', ha='center', va='center', transform=ax.transAxes)
        return

    df = pd.read_csv(delta_csv_path)
    df = rename_genotype(df)
    sub = df[(df['Drug'] == drug_name) & (df['Pathway'] == pathway_name)].copy()

    if sub.empty:
        ax.text(0.5, 0.5, f'No {drug_name}\n{pathway_name} data', ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    # User says Post - Pre is correct for data, but magnitude decrease should "go down"
    # Magnitude = -Area (since GABAB areas are negative)
    # Mag_Delta = Mag_Post - Mag_Pre = (-sub['Post_GABAB_Area']) - (-sub['Pre_GABAB_Area'])
    # This equals Pre_Area - Post_Area.
    sub['Inhibition_Delta'] = sub['Pre_GABAB_Area'] - sub['Post_GABAB_Area']

    plot_bar_scatter(ax, sub, 'Genotype', 'Inhibition_Delta', 'Genotype',
                     order=['WT', 'I80T/+'], unique_col='Cell_ID')
    ax.set_ylabel('Δ GABAB Area (mV·s)', fontsize=7)
    pathway_label = 'ECIII' if pathway_name == 'Perforant' else 'CA3 Apical'
    ax.set_title(f'{drug_name} Δ {pathway_label}', fontsize=8, fontweight='bold')
    ax.set_box_aspect(1)

    # Statistical annotation
    if df_stats is not None:
        comparison = f"WT vs GNB1 ({pathway_name} Unitary Delta)"
        row = df_stats[(df_stats['Drug'] == drug_name) & (df_stats['Comparison'] == comparison)]
        if not row.empty:
            sig = row.iloc[0].get('Significance', 'ns')
            wt_vals = sub[sub['Genotype'] == 'WT']['Inhibition_Delta'].dropna()
            gnb1_vals = sub[sub['Genotype'] == 'I80T/+']['Inhibition_Delta'].dropna()
            if len(wt_vals) > 0 and len(gnb1_vals) > 0:
                m_wt, m_gnb = wt_vals.mean(), gnb1_vals.mean()
                s_wt, s_gnb = wt_vals.sem(), gnb1_vals.sem()
                
                # Position stars above or below depending on sign
                if m_wt >= 0 and m_gnb >= 0:
                    y_pos = max(m_wt + s_wt, m_gnb + s_gnb) * 1.15 + 0.05
                elif m_wt <= 0 and m_gnb <= 0:
                    y_pos = min(m_wt - s_wt, m_gnb - s_gnb) * 1.15 - 0.05
                else:
                    y_pos = max(abs(m_wt)+s_wt, abs(m_gnb)+s_gnb) * 1.15 + 0.05
                
                ax.text(0.5, y_pos, sig, ha='center', va='bottom' if y_pos > 0 else 'top', 
                        fontsize=9, fontweight='bold', transform=ax.get_xaxis_transform())



def plot_plateau_girk_delta(ax, plateau_csv_path, drug_name, label, df_stats=None):
    """
    Plot Delta Plateau Area for GIRK pharmacology (ML297 or ETX).
    """
    add_subplot_label(ax, label)
    if os.path.exists(plateau_csv_path):
        df = pd.read_csv(plateau_csv_path)
        df_sub = df[df['Drug'] == drug_name].copy()
        df_sub = rename_genotype(df_sub)
        
        # IMPORTANT: Filter for 'Both' pathway to avoid multiplying N counts
        df_sub = df_sub[df_sub['Pathway'] == 'Both'].copy()
        
        plot_bar_scatter(ax, df_sub, 'Genotype', 'Delta_Area', 'Genotype', order=['WT', 'I80T/+'], unique_col='Cell_ID')
        ax.set_ylabel('Δ Plateau Area (mV-s)')
        ax.set_title(f'GIRK Effect: {drug_name}', fontsize=8, fontweight='bold')
        ax.set_box_aspect(1)

        if df_stats is not None:
             comparison = "WT vs GNB1 (Plateau Delta)"
             # For Plateau Delta, we use 'Both' pathway for the main summary bar
             row = df_stats[(df_stats['Drug'] == drug_name) & 
                            (df_stats['Comparison'] == comparison) &
                            (df_stats['Pathway'] == 'Both')]
             if not row.empty:
                 sig = row.iloc[0].get('Significance', 'ns')
                 y_max = max(df_sub.groupby('Genotype')['Delta_Area'].mean())
                 y_pos = y_max + (abs(y_max) * 0.1)
                 ax.text(0.5, y_pos, sig, ha='center', va='bottom', fontsize=9, fontweight='bold')
                 if sig != 'ns':
                     ax.plot([0, 1], [y_pos, y_pos], 'k-', lw=0.8)
    else:
        ax.text(0.5, 0.5, 'No Plateau Data', ha='center', va='center')
    
