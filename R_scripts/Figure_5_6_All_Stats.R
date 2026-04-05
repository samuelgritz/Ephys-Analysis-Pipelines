#!/usr/bin/env Rscript
################################################################################
# Figure 5 & 6 - All Stats with FDR Correction
# 
# This script performs all statistical analyses for Figures 5 & 6:
#   - Excitation Amplitude (Gabazine EPSP) — WT vs GNB1 per pathway
#   - Inhibition (GABAA) Amplitude — WT vs GNB1 per pathway
#   - GABAB Inhibition Area — WT vs GNB1 per pathway
#   - Gabazine Supralinearity — WT vs GNB1 per pathway
#   - E:I Imbalance — WT vs GNB1 per pathway
#
# STATISTICAL MODEL:
#   Linear Mixed Effects (LME) with Subject as random effect:
#     response ~ Genotype * ISI_Time + (1 | Subject)
#
# FDR CORRECTION LOGIC:
#   For each ANALYSIS (e.g. Excitation) × PATHWAY (e.g. Perforant):
#     1. Check if the Genotype:ISI_Time INTERACTION EFFECT is significant (p < 0.05)
#     2. If YES: Pool the 5 post-hoc p-values (ISI 300, 100, 50, 25, 10) and
#        apply Benjamini-Hochberg FDR correction WITHIN that pathway only
#     3. If NO: No post-hoc tests are run (no p-values to correct)
#
#   IMPORTANT: FDR correction is NEVER pooled across pathways.
#   Each pathway's 5 ISI p-values are corrected independently.
#
# OUTPUTS:
#   - Figure_5_6_All_Stats_ANOVA.csv          - All ANOVA/LME results
#   - Figure_5_6_All_Stats_Uncorrected.csv    - All uncorrected post-hoc p-values  
#   - Figure_5_6_All_Stats_FDR_Corrected.csv  - FDR-corrected p-values by group
#   - Figure_5_6_Significance_Markers.csv     - Markers for plotting (# main, * post-hoc)
################################################################################

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)

################################################################################
# 1. LOAD AND PREPARE DATA
################################################################################

cat("\n==============================================================================\n")
cat("FIGURE 5 & 6: ALL STATS WITH FDR CORRECTION\n")
cat("==============================================================================\n\n")

# -----------------------------------------------------------------------------
# Determine base path — works from project root or R_scripts/ subdirectory
# -----------------------------------------------------------------------------
if (dir.exists("paper_data")) {
  base_path <- "paper_data/E_I_data/"
} else if (dir.exists("../paper_data")) {
  base_path <- "../paper_data/E_I_data/"
} else {
  stop("ERROR: 'paper_data' directory not found. Please run from project root or R_scripts directory.")
}

# -----------------------------------------------------------------------------
# Load EPSP amplitude data (wide format: ISI10..ISI300 columns)
# Used for: Excitation (Gabazine), Control amplitude, and Drug effect analyses
# -----------------------------------------------------------------------------
file_name <- paste0(base_path, 'E_I_EPSP_amplitudes_R_format.csv')

if (!file.exists(file_name)) {
  stop("ERROR: Data file not found: ", file_name, "\n",
       "Please run Analyze_and_Export_E_I_data.py first to generate data.")
}

E_I_experiment <- read.csv(file_name)
cat("✓ Loaded EPSP amplitude data from:", file_name, "\n")
cat("  Rows:", nrow(E_I_experiment), "\n\n")

# ISI columns present in the wide-format data
ISI_cols <- c("ISI10", "ISI25", "ISI50", "ISI100", "ISI300")

# Clean: remove rows with any missing ISI values (needed for complete RM design)
E_I_experiment_clean <- E_I_experiment %>%
  filter(complete.cases(!!!syms(ISI_cols)))

# Pivot to long format for LME analysis:
#   Each row = one Subject × ISI_Time observation
E_I_experiment_long <- E_I_experiment_clean %>%
  pivot_longer(
    cols = all_of(ISI_cols),
    names_to = "ISI_Time",
    values_to = "EPSP_Amplitude"
  ) %>%
  mutate(
    Genotype = as.factor(Genotype),
    Drug = factor(Drug, levels = c(0, 1), labels = c("Control", "Gabazine")),
    Pathway = as.factor(Pathway),
    Subject = as.factor(Subject),
    # ISI_Time is ordered: ISI10, ISI25, ISI50, ISI100, ISI300
    ISI_Time = factor(ISI_Time, levels = ISI_cols)
  )

cat("  Converted to long format:", nrow(E_I_experiment_long), "rows\n")
cat("  Subjects (cells):", length(unique(E_I_experiment_long$Subject)), "\n\n")

# -----------------------------------------------------------------------------
# Load E_I amplitudes data (already long-ish format)
# Used for: Supralinearity and E:I Imbalance analyses
# -----------------------------------------------------------------------------
file_name_ei <- paste0(base_path, 'E_I_amplitudes.csv')

if (!file.exists(file_name_ei)) {
  stop("ERROR: Data file not found: ", file_name_ei, "\n",
       "Please run Analyze_and_Export_E_I_data.py first to generate data.")
}

E_I_data <- read.csv(file_name_ei)
cat("✓ Loaded Supralinearity and E:I Imbalance data from:", file_name_ei, "\n")
cat("  Rows:", nrow(E_I_data), "\n\n")

# Prepare factors for E:I data
E_I_clean <- E_I_data %>%
  mutate(
    Subject = as.factor(Cell_ID),
    Genotype = as.factor(Genotype),
    Pathway = as.factor(Pathway),
    # Convert numeric ISI column to factor matching ISI_cols naming
    ISI_Time = as.factor(paste0("ISI", ISI))
  )

################################################################################
# 2. INITIALIZE STORAGE LISTS
################################################################################

# Storage for ANOVA/LME omnibus test results
all_anova_results <- list()

# Storage for post-hoc pairwise comparison p-values (uncorrected)
all_uncorrected_pvalues <- list()

# Storage for N counts per analysis/pathway
summary_data <- list()

# Map pathway codes (1, 2, 3) to descriptive names
pathway_names <- c("1" = "Perforant", "2" = "Schaffer", "3" = "Basal_Stratum_Oriens")

################################################################################
# 3. HELPER FUNCTION: Run LME and Extract ANOVA + Post-hoc Results
#
# This function:
#   1. Fits a linear mixed effects model (LME) with Subject as random effect
#   2. Extracts the ANOVA table (Type III tests)
#   3. Checks for significant INTERACTION EFFECT (Genotype:ISI_Time)
#   4. If interaction is significant: runs post-hoc pairwise comparisons
#      (emmeans) at each ISI level, WITHOUT any correction
#      (correction is applied later during the FDR step)
#   5. Returns both the ANOVA table and the uncorrected post-hoc p-values
#
# IMPORTANT: Post-hoc tests are ONLY run when the interaction is significant.
#   This is the gate that controls whether FDR correction is applied downstream.
#
# Parameters:
#   data            - Dataframe (long format) filtered to one pathway
#   formula_str     - LME formula string (e.g. "Y ~ Genotype * ISI_Time + (1|Subject)")
#   analysis_name   - Column name of the response variable (e.g. "EPSP_Amplitude")
#   pathway_name    - Name of the pathway (e.g. "Perforant")
#   comparison_name - Description of the comparison (e.g. "WT_vs_GNB1_Gabazine")
#   fdr_group       - FDR grouping key: format is "Analysis_PathwayAbbrev"
#                     e.g. "Exc_Pe" = Excitation × Perforant
#                     FDR correction pools the 5 ISI p-values WITHIN this group only
#   effect_to_test  - Which ANOVA term gates post-hoc: "interaction" (default)
#   analysis_label  - Label for output CSVs (defaults to analysis_name)
################################################################################

run_lmer_analysis <- function(data, formula_str, analysis_name, pathway_name, 
                               comparison_name, fdr_group, effect_to_test = "interaction",
                               analysis_label = analysis_name) {
  
  # -------------------------------------------------------------------------
  # Step 1: Data cleaning — keep only columns needed for this model
  # -------------------------------------------------------------------------
  model_cols <- c("Subject", "Genotype", "Drug", "ISI_Time", analysis_name)
  model_data <- data %>%
    select(all_of(model_cols[model_cols %in% names(data)])) %>%
    drop_na(all_of(analysis_name))
  
  # -------------------------------------------------------------------------
  # Step 2: Fit LME model
  # Subject = random intercept (repeated measures within each cell)
  # -------------------------------------------------------------------------
  model <- tryCatch({
    lmer(as.formula(formula_str), data = model_data, REML = TRUE)
  }, error = function(e) {
    cat("    Error fitting model:", conditionMessage(e), "\n")
    return(NULL)
  })
  
  if (is.null(model)) return(list(anova = NULL, posthoc = NULL))
  
  # -------------------------------------------------------------------------
  # Step 3: Extract ANOVA table (Type III via lmerTest::anova)
  # This gives F-tests for:
  #   - Genotype (main effect)
  #   - ISI_Time (main effect)
  #   - Genotype:ISI_Time (interaction effect)
  # -------------------------------------------------------------------------
  anova_result <- anova(model)
  anova_df <- as.data.frame(anova_result)
  anova_df$Effect <- rownames(anova_df)
  colnames(anova_df)[colnames(anova_df) == "Pr(>F)"] <- "P_Value"
  anova_df$Analysis <- analysis_label
  anova_df$Pathway <- pathway_name
  anova_df$Comparison <- comparison_name
  
  # -------------------------------------------------------------------------
  # Step 4: Calculate group-level summary statistics (Mean ± SEM)
  # for inclusion in output CSVs
  # -------------------------------------------------------------------------
  group_col <- if (grepl("Genotype", formula_str)) "Genotype" else "Drug"
  
  summary_stats <- model_data %>%
    group_by(!!sym(group_col)) %>%
    summarise(
      Mean = mean(!!sym(analysis_name), na.rm = TRUE),
      SEM = sd(!!sym(analysis_name), na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    pivot_wider(
      names_from = !!sym(group_col),
      values_from = c(Mean, SEM)
    )
  
  # Merge summary stats into every row of the ANOVA output
  anova_df <- cbind(anova_df, summary_stats)
  anova_df$Significant <- ifelse(anova_df$P_Value < 0.05, "*", "ns")
  
  # -------------------------------------------------------------------------
  # Step 5: Determine INTERACTION and MAIN EFFECT p-values
  # The interaction term is the one containing ":" (e.g. "Genotype:ISI_Time")
  # -------------------------------------------------------------------------
  if (effect_to_test == "interaction") {
    # Find the interaction term (contains ":")
    interaction_rows <- grep(":", anova_df$Effect)
    if (length(interaction_rows) > 0) {
      p_val <- anova_df$P_Value[interaction_rows[1]]  # Interaction p
      main_p <- anova_df$P_Value[1]                    # First main effect p (Genotype)
    } else {
      p_val <- NA
      main_p <- anova_df$P_Value[1]
    }
  } else {
    p_val <- anova_df$P_Value[1]
    main_p <- p_val
  }
  
  # -------------------------------------------------------------------------
  # Step 6: Post-hoc pairwise comparisons (emmeans)
  # ONLY RUN IF INTERACTION IS SIGNIFICANT (p < 0.05)
  #
  # This is the critical gate:
  #   - If Genotype:ISI_Time interaction is significant → run post-hoc
  #     contrasts at each ISI level (WT vs GNB1 | ISI10, ISI25, ISI50, ISI100, ISI300)
  #   - Post-hoc p-values are stored UNCORRECTED here
  #   - FDR correction is applied later in Section 9, pooling the 5 ISI
  #     p-values WITHIN this specific Analysis × Pathway combination
  #   - If interaction is NOT significant → no post-hoc tests, no FDR correction
  # -------------------------------------------------------------------------
  posthoc_list <- list()
  
  # Count subjects for quality check
  n_wt <- length(unique(model_data$Subject[model_data$Genotype == "WT"]))
  n_gnb1 <- length(unique(model_data$Subject[model_data$Genotype == "GNB1"]))
  
  # Gate: ONLY run post-hoc if INTERACTION is significant
  if (!is.na(p_val) && p_val < 0.05) {
    # Determine the correct emmeans contrast based on the model formula
    if (grepl("Genotype", formula_str) && grepl("ISI_Time", formula_str)) {
      # Compare Genotype levels (WT vs GNB1) at each ISI level
      emm <- emmeans(model, pairwise ~ Genotype | ISI_Time)
    } else if (grepl("Drug", formula_str) && grepl("ISI_Time", formula_str)) {
      # Compare Drug levels (Control vs Gabazine) at each ISI level
      emm <- emmeans(model, pairwise ~ Drug | ISI_Time)
    } else {
      emm <- NULL
    }
    
    if (!is.null(emm)) {
      # Extract contrasts WITHOUT any adjustment (adjust = "none")
      # FDR correction will be applied later across the 5 ISIs for this pathway
      contrasts <- summary(emm$contrasts, adjust = "none")
      
      # Store each ISI-level contrast as a separate row
      for (i in 1:nrow(contrasts)) {
        posthoc_list[[length(posthoc_list) + 1]] <- list(
          Analysis = analysis_label,        # e.g. "Gabazine_Amplitude"
          Pathway = pathway_name,           # e.g. "Perforant"
          Comparison = comparison_name,     # e.g. "WT_vs_GNB1_Gabazine"
          ISI = as.character(contrasts$ISI_Time[i]),  # e.g. "ISI10"
          estimate = contrasts$estimate[i], # Mean difference
          SE = contrasts$SE[i],             # Standard error of difference
          df = contrasts$df[i],             # Degrees of freedom
          t_ratio = contrasts$t.ratio[i],   # t-statistic
          p_value_uncorrected = contrasts$p.value[i],  # Raw p-value (NO correction)
          # FDR_Group = key for downstream FDR correction
          # Format: "AnalysisAbbrev_PathwayAbbrev"
          # FDR correction will pool the 5 ISI p-values within this group ONLY
          FDR_Group = fdr_group,
          Main_Effect_p = main_p,           # Genotype main effect p
          Interaction_p = p_val             # Genotype:ISI_Time interaction p
        )
      }
    }
  }
  
  # -------------------------------------------------------------------------
  # Step 7: FALLBACK for low-N situations
  # If the LME dropped subjects (N < 3 per group), run per-ISI Welch t-tests
  # as a safety net. This captures ISI 300 (unitary) significance even when
  # other ISIs are missing data.
  # -------------------------------------------------------------------------
  if (n_wt < 3 || n_gnb1 < 3) {
    cat("    Warning: Low N in LME for", analysis_name, "(WT:", n_wt, "GNB1:", n_gnb1, "). Running per-ISI fallback.\n")
    unique_isis <- unique(model_data$ISI_Time)
    
    for (isi in unique_isis) {
      isi_data <- model_data[model_data$ISI_Time == isi, ]
      wt_vals <- isi_data[[analysis_name]][isi_data$Genotype == "WT"]
      gnb1_vals <- isi_data[[analysis_name]][isi_data$Genotype == "GNB1"]
      
      if (length(wt_vals) >= 3 && length(gnb1_vals) >= 3) {
        t_res <- t.test(wt_vals, gnb1_vals)
        posthoc_list[[length(posthoc_list) + 1]] <- list(
          Analysis = analysis_label,
          Pathway = pathway_name,
          Comparison = comparison_name,
          ISI = as.character(isi),
          estimate = mean(gnb1_vals) - mean(wt_vals),
          SE = t_res$stderr,
          df = t_res$parameter,
          t_ratio = t_res$statistic,
          p_value_uncorrected = t_res$p.value,
          FDR_Group = fdr_group,
          Main_Effect_p = main_p,
          Interaction_p = p_val
        )
      }
    }
  }
  
  return(list(anova = anova_df, posthoc = posthoc_list, main_p = main_p, interaction_p = p_val))
}

################################################################################
# 4. EXCITATION AMPLITUDE ANALYSES — ALL PATHWAYS
#
# Analysis: Gabazine EPSP Amplitude (excitatory component after GABAA block)
# Model:    Gabazine_Amplitude ~ Genotype * ISI_Time + (1 | Subject)
# Question: Does excitation differ between WT and GNB1 across ISIs?
#
# Also runs Control amplitude and within-genotype Drug effect analyses.
################################################################################

cat("==============================================================================\n")
cat("RUNNING EPSP AMPLITUDE ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_code in c("1", "2", "3")) {
  pathway_name <- pathway_names[pathway_code]
  cat("\n--- ", pathway_name, " PATHWAY ---\n\n", sep = "")
  
  # Count cells (unique subjects) for this pathway
  pathway_data <- E_I_experiment_long %>% filter(Pathway == pathway_code)
  n_wt <- length(unique(pathway_data$Subject[pathway_data$Genotype == "WT"]))
  n_gnb1 <- length(unique(pathway_data$Subject[pathway_data$Genotype == "GNB1"]))
  
  # Store N counts for summary output
  summary_data[[paste(pathway_name, "Gabazine_Amplitude", sep = "_")]] <- list(
    Pathway = pathway_name,
    Analysis = "Gabazine_Amplitude",
    N_WT = n_wt,
    N_GNB1 = n_gnb1
  )
  
  # Analysis 4a: WT vs GNB1 — Gabazine condition only (EXCITATION)
  # This is the primary excitation analysis for Figures 5 & 6
  # FDR Group: "Exc_Pe" / "Exc_Sc" / "Exc_Ba" — one per pathway
  # -----------------------------------------------------------------------
  gabazine_data <- E_I_experiment_long %>% 
    filter(Pathway == pathway_code & Drug == "Gabazine") %>%
    rename(Gabazine_Amplitude = EPSP_Amplitude)
    
  result_gab <- run_lmer_analysis(
    data = gabazine_data,
    formula_str = "Gabazine_Amplitude ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "Gabazine_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1",
    # FDR group: pools 5 ISIs for THIS pathway only
    fdr_group = paste("Gabazine_Amplitude", pathway_name, sep = " x "),
    analysis_label = "Gabazine_Amplitude"
  )
  
  if (!is.null(result_gab$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_gab$anova
    cat("  WT vs GNB1 (Gabazine): Interaction p =", 
        ifelse(is.na(result_gab$interaction_p), "NA", round(result_gab$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_gab$posthoc)
  
  # -----------------------------------------------------------------------
  # Analysis 4b: WT vs GNB1 — Control condition only
  # -----------------------------------------------------------------------
  control_data <- E_I_experiment_long %>% 
    filter(Pathway == pathway_code & Drug == "Control") %>%
    rename(Control_Amplitude = EPSP_Amplitude)
    
  result_ctrl <- run_lmer_analysis(
    data = control_data,
    formula_str = "Control_Amplitude ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "Control_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1_Control",
    fdr_group = paste("Control_Amplitude", pathway_name, sep = " x "),
    analysis_label = "Control_Amplitude"
  )
  
  if (!is.null(result_ctrl$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_ctrl$anova
    cat("  WT vs GNB1 (Control): Interaction p =", 
        ifelse(is.na(result_ctrl$interaction_p), "NA", round(result_ctrl$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_ctrl$posthoc)
  
  # -----------------------------------------------------------------------
  # Analysis 4c: WT — Control vs Gabazine (within-genotype drug effect)
  # -----------------------------------------------------------------------
  wt_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Genotype == "WT")
  result_wt <- run_lmer_analysis(
    data = wt_data,
    formula_str = "EPSP_Amplitude ~ Drug + Drug * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_Control_vs_Gabazine",
    fdr_group = paste("WT_Control_vs_Gabazine", pathway_name, sep = " x "),
    analysis_label = "Control_vs_Gabazine_Amplitude"
  )
  
  if (!is.null(result_wt$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_wt$anova
    cat("  WT Control vs Gabazine: Interaction p =", 
        ifelse(is.na(result_wt$interaction_p), "NA", round(result_wt$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_wt$posthoc)
  
  # -----------------------------------------------------------------------
  # Analysis 4d: GNB1 — Control vs Gabazine (within-genotype drug effect)
  # -----------------------------------------------------------------------
  gnb1_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Genotype == "GNB1")
  result_gnb1 <- run_lmer_analysis(
    data = gnb1_data,
    formula_str = "EPSP_Amplitude ~ Drug + Drug * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "GNB1_Control_vs_Gabazine",
    fdr_group = paste("GNB1_Control_vs_Gabazine", pathway_name, sep = " x "),
    analysis_label = "Control_vs_Gabazine_Amplitude"
  )
  
  if (!is.null(result_gnb1$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_gnb1$anova
    cat("  GNB1 Control vs Gabazine: Interaction p =", 
        ifelse(is.na(result_gnb1$interaction_p), "NA", round(result_gnb1$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_gnb1$posthoc)
}

################################################################################
# 5. GABAZINE SUPRALINEARITY ANALYSES
#
# Analysis: Gabazine_Supralinearity = Gabazine_EPSP - Control_EPSP (% difference)
# Model:    Gabazine_Supralinearity ~ Genotype * ISI_Time + (1 | Subject)
# Question: Does the supralinear boost differ between WT and GNB1 across ISIs?
# FDR:      Pool 5 ISI p-values per pathway, correct independently
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING GABAZINE SUPRALINEARITY ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
  cat("--- ", pathway_name, " ---\n", sep = "")
  
  # Filter to cells with valid supralinearity data for this pathway
  supra_data <- E_I_clean %>% 
    filter(Pathway == pathway_name & !is.na(Gabazine_Supralinearity))
  
  if (nrow(supra_data) == 0) {
    cat("  No data available\n")
    next
  }
  
  # Count cells for summary
  n_wt <- length(unique(supra_data$Subject[supra_data$Genotype == "WT"]))
  n_gnb1 <- length(unique(supra_data$Subject[supra_data$Genotype == "GNB1"]))
  
  summary_data[[paste(pathway_name, "Gabazine_Supralinearity", sep = "_")]] <- list(
    Pathway = pathway_name,
    Analysis = "Gabazine_Supralinearity",
    N_WT = n_wt,
    N_GNB1 = n_gnb1
  )
  
  result <- run_lmer_analysis(
    data = supra_data,
    formula_str = "Gabazine_Supralinearity ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "Gabazine_Supralinearity",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1",
    # FDR group: pools 5 ISIs for THIS pathway only
    fdr_group = paste("Gabazine_Supralinearity", pathway_name, sep = " x ")
  )
  
  if (!is.null(result$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result$anova
    cat("  Genotype:ISI_Time p =", 
        ifelse(is.na(result$interaction_p), "NA", round(result$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result$posthoc)
}

################################################################################
# 6. E:I IMBALANCE ANALYSES
#
# Analysis: E_I_Imbalance = Excitation / (Excitation + Inhibition)
# Model:    E_I_Imbalance ~ Genotype * ISI_Time + (1 | Subject)
# Question: Does the E:I balance shift between WT and GNB1 across ISIs?
# FDR:      Pool 5 ISI p-values per pathway, correct independently
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING E:I IMBALANCE ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
  cat("--- ", pathway_name, " ---\n", sep = "")
  
  # Filter to cells with valid E:I data for this pathway
  ei_data <- E_I_clean %>% 
    filter(Pathway == pathway_name & !is.na(E_I_Imbalance))
  
  if (nrow(ei_data) == 0) {
    cat("  No data available\n")
    next
  }
  
  # Count cells for summary
  n_wt <- length(unique(ei_data$Subject[ei_data$Genotype == "WT"]))
  n_gnb1 <- length(unique(ei_data$Subject[ei_data$Genotype == "GNB1"]))
  
  summary_data[[paste(pathway_name, "E_I_Imbalance", sep = "_")]] <- list(
    Pathway = pathway_name,
    Analysis = "E_I_Imbalance",
    N_WT = n_wt,
    N_GNB1 = n_gnb1
  )
  
  result <- run_lmer_analysis(
    data = ei_data,
    formula_str = "E_I_Imbalance ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "E_I_Imbalance",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1",
    # FDR group: pools 5 ISIs for THIS pathway only
    fdr_group = paste("E_I_Imbalance", pathway_name, sep = " x ")
  )
  
  if (!is.null(result$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result$anova
    cat("  Genotype:ISI_Time p =", 
        ifelse(is.na(result$interaction_p), "NA", round(result$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result$posthoc)
}

################################################################################
# 7. GABAA INHIBITION ANALYSES
#
# Analysis: GABAA Inhibition Amplitude (Control - Gabazine, negative = inhibition)
# Model:    Inhibition_Amplitude ~ Genotype * ISI_Time + (1 | Subject)
# Question: Does GABAA-mediated inhibition differ between WT and GNB1?
# FDR:      Pool 5 ISI p-values per pathway, correct independently
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING GABAA INHIBITION ANALYSES\n")
cat("==============================================================================\n\n")

# Load GABAA inhibition data (wide format: ISI10..ISI300 columns)
file_name_inh <- paste0(base_path, 'E_I_GABAA_Inhibition_R_format.csv')

if (file.exists(file_name_inh)) {
  inh_data <- read.csv(file_name_inh)
  cat("✓ Loaded GABAA Inhibition data from:", file_name_inh, "\n")
  
  # Pivot to long format (same structure as excitation data)
  inh_clean <- inh_data %>%
    filter(complete.cases(!!!syms(ISI_cols))) %>%
    pivot_longer(
      cols = all_of(ISI_cols),
      names_to = "ISI_Time",
      values_to = "Inhibition_Amplitude"
    ) %>%
    mutate(
      Genotype = as.factor(Genotype),
      Pathway = factor(Pathway, levels = c(1, 2, 3), labels = c("Perforant", "Schaffer", "Basal_Stratum_Oriens")),
      Subject = as.factor(Subject),
      ISI_Time = factor(ISI_Time, levels = ISI_cols)
    )

  for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
    cat("--- ", pathway_name, " ---\n", sep = "")
    
    subset_data <- inh_clean %>% filter(Pathway == pathway_name)
    if (nrow(subset_data) == 0) {
      cat("  No data available\n")
      next
    }
    
    # Count cells for summary
    n_wt <- length(unique(subset_data$Subject[subset_data$Genotype == "WT"]))
    n_gnb1 <- length(unique(subset_data$Subject[subset_data$Genotype == "GNB1"]))
    
    summary_data[[paste(pathway_name, "Inhibition_Amplitude", sep = "_")]] <- list(
      Pathway = pathway_name,
      Analysis = "Inhibition_Amplitude",
      N_WT = n_wt,
      N_GNB1 = n_gnb1
    )
    
    result <- run_lmer_analysis(
      data = subset_data,
      formula_str = "Inhibition_Amplitude ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
      analysis_name = "Inhibition_Amplitude",
      pathway_name = pathway_name,
      comparison_name = "WT_vs_GNB1",
      # FDR group: pools 5 ISIs for THIS pathway only
      fdr_group = paste("Inhibition_Amplitude", pathway_name, sep = " x ")
    )
    
    if (!is.null(result$anova)) {
      all_anova_results[[length(all_anova_results) + 1]] <- result$anova
      cat("  Genotype:ISI_Time p =", 
          ifelse(is.na(result$interaction_p), "NA", round(result$interaction_p, 4)), "\n")
    }
    all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result$posthoc)
  }
} else {
  cat("⚠ GABAA Inhibition data not found\n")
}

################################################################################
# 8. GABAB AREA ANALYSES
#
# Analysis: GABAB Area = integral of negative-going (below zero) trace component
#           measured from the Gabazine trace (after GABAA block)
# Model:    GABAB_Area ~ Genotype * ISI_Time + (1 | Subject)
# Question: Does GABAB-mediated inhibition area differ between WT and GNB1?
# FDR:      Pool 5 ISI p-values per pathway, correct independently
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING GABAB AREA ANALYSES\n")
cat("==============================================================================\n\n")

# Load GABAB area data (wide format: ISI10..ISI300 columns)
file_name_gabab <- paste0(base_path, 'E_I_GABAB_Area_R_format.csv')

if (file.exists(file_name_gabab)) {
  gabab_data <- read.csv(file_name_gabab)
  cat("✓ Loaded GABAB Area data from:", file_name_gabab, "\n")
  
  # Pivot to long format
  gabab_clean <- gabab_data %>%
    filter(complete.cases(!!!syms(ISI_cols))) %>%
    pivot_longer(
      cols = all_of(ISI_cols),
      names_to = "ISI_Time",
      values_to = "GABAB_Area"
    ) %>%
    mutate(
      Genotype = as.factor(Genotype),
      Pathway = factor(Pathway, levels = c(1, 2, 3), labels = c("Perforant", "Schaffer", "Basal_Stratum_Oriens")),
      Subject = as.factor(Subject),
      ISI_Time = factor(ISI_Time, levels = ISI_cols)
    )

  for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
    cat("--- ", pathway_name, " ---\n", sep = "")
    
    subset_data <- gabab_clean %>% filter(Pathway == pathway_name)
    if (nrow(subset_data) == 0) {
      cat("  No data available\n")
      next
    }
    
    # Count cells for summary
    n_wt <- length(unique(subset_data$Subject[subset_data$Genotype == "WT"]))
    n_gnb1 <- length(unique(subset_data$Subject[subset_data$Genotype == "GNB1"]))
    
    summary_data[[paste(pathway_name, "GABAB_Area", sep = "_")]] <- list(
      Pathway = pathway_name,
      Analysis = "GABAB_Area",
      N_WT = n_wt,
      N_GNB1 = n_gnb1
    )
    
    result <- run_lmer_analysis(
      data = subset_data,
      formula_str = "GABAB_Area ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
      analysis_name = "GABAB_Area",
      pathway_name = pathway_name,
      comparison_name = "WT_vs_GNB1",
      # FDR group: pools 5 ISIs for THIS pathway only
      fdr_group = paste("GABAB_Area", pathway_name, sep = " x ")
    )
    
    if (!is.null(result$anova)) {
      all_anova_results[[length(all_anova_results) + 1]] <- result$anova
      cat("  Genotype:ISI_Time p =", 
          ifelse(is.na(result$interaction_p), "NA", round(result$interaction_p, 4)), "\n")
    }
    all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result$posthoc)
  }
} else {
  cat("⚠ GABAB Area data not found\n")
}

################################################################################
# 9. COMPILE AND SAVE ANOVA RESULTS
################################################################################

cat("\n==============================================================================\n")
cat("SAVING ANOVA RESULTS\n")
cat("==============================================================================\n\n")

# Combine all ANOVA results into a single dataframe
all_anova_df <- bind_rows(all_anova_results) %>%
  select(Analysis, Pathway, Comparison, Effect, 
         starts_with("Mean_"), starts_with("SEM_"),
         `Sum Sq`, `Mean Sq`, NumDF, DenDF, `F value`, P_Value, Significant) %>%
  arrange(Analysis, Pathway, Comparison, Effect)

# Save with Figure_5_6_ prefix
output_anova <- paste0(base_path, 'Figure_5_6_All_Stats_ANOVA.csv')
write.csv(all_anova_df, output_anova, row.names = FALSE)
cat("✓ Saved ANOVA results to:", output_anova, "\n")
cat("  Total ANOVA tests:", nrow(all_anova_df), "\n\n")

################################################################################
# 10. COMPILE AND SAVE UNCORRECTED P-VALUES
################################################################################

cat("==============================================================================\n")
cat("COMPILING UNCORRECTED P-VALUES\n")
cat("==============================================================================\n\n")

# Convert list of post-hoc results to a single dataframe
all_pvalues_df <- bind_rows(all_uncorrected_pvalues)

cat("Total uncorrected p-values collected:", nrow(all_pvalues_df), "\n")
if (nrow(all_pvalues_df) > 0) {
  cat("By FDR Group:\n")
  print(table(all_pvalues_df$FDR_Group))
}
cat("\n")

# Save uncorrected p-values with Figure_5_6_ prefix
output_uncorrected <- paste0(base_path, 'Figure_5_6_All_Stats_Uncorrected.csv')
write.csv(all_pvalues_df, output_uncorrected, row.names = FALSE)
cat("✓ Saved uncorrected p-values to:", output_uncorrected, "\n\n")

################################################################################
# 11. APPLY FDR CORRECTION BY GROUPS
#
# FDR CORRECTION LOGIC:
#   - Each FDR_Group = one Analysis × one Pathway (e.g. "Exc_Pe" = Excitation × Perforant)
#   - Each group contains exactly 5 p-values (one per ISI: 300, 100, 50, 25, 10)
#   - FDR (Benjamini-Hochberg) correction is applied WITHIN each group independently
#   - p-values are NEVER pooled across different pathways
#   - p-values are NEVER pooled across different analyses
#
# IMPORTANT: Post-hoc p-values only exist for groups where the INTERACTION
#   EFFECT (Genotype:ISI_Time) was significant in the ANOVA/LME.
#   If the interaction was NOT significant, no post-hoc tests were run,
#   so there are no p-values to correct for that group.
#
# Example:
#   If Excitation × Perforant has significant interaction:
#     → Pool ISI300, ISI100, ISI50, ISI25, ISI10 p-values
#     → Apply p.adjust(method = "fdr") across these 5 p-values
#     → Report FDR-corrected p-values
#   If Excitation × Schaffer has NO significant interaction:
#     → No post-hoc tests were run → no FDR correction needed
################################################################################

cat("==============================================================================\n")
cat("APPLYING FDR CORRECTION BY GROUPS\n")
cat("==============================================================================\n\n")

if (nrow(all_pvalues_df) > 0) {
  # Initialize FDR-corrected column
  all_pvalues_df$p_value_FDR <- NA
  
  # Get all unique FDR groups (each = one Analysis × one Pathway)
  groups <- unique(all_pvalues_df$FDR_Group)
  
  for (grp in groups) {
    # Get the row indices belonging to this FDR group
    idx <- which(all_pvalues_df$FDR_Group == grp)
    
    if (length(idx) > 0) {
      # Extract the interaction p-value for this specific group (Analysis x Pathway)
      interaction_p <- unique(all_pvalues_df$Interaction_p[idx])[1]
      
      # -----------------------------------------------------------------------
      # RULE: Only FDR-correct the 5 ISI p-values if the interaction effect 
      # (Genotype:ISI_Time) is significant (p < 0.05) for that pathway.
      # -----------------------------------------------------------------------
      if (!is.na(interaction_p) && interaction_p < 0.05) {
        # Apply Benjamini-Hochberg FDR correction to the 5 ISI p-values
        # in this group ONLY (never across pathways or analyses)
        all_pvalues_df$p_value_FDR[idx] <- p.adjust(
          all_pvalues_df$p_value_uncorrected[idx], 
          method = "fdr"
        )
        cat("✓ FDR corrected group:", grp, "(interaction p =", round(interaction_p, 4), ")\n")
      } else {
        # If interaction is not significant, do not apply FDR correction.
        # We leave p_value_FDR as NA so it doesn't get significance markers (*).
        all_pvalues_df$p_value_FDR[idx] <- NA
        cat("x Skipping FDR correction (interaction not significant):", grp, "\n")
      }
    }
  }
  
  # -------------------------------------------------------------------------
  # Add significance star indicators for both uncorrected and FDR-corrected
  # -------------------------------------------------------------------------
  all_pvalues_df$Significant_Uncorrected <- case_when(
    all_pvalues_df$p_value_uncorrected < 0.001 ~ "***",
    all_pvalues_df$p_value_uncorrected < 0.01 ~ "**",
    all_pvalues_df$p_value_uncorrected < 0.05 ~ "*",
    TRUE ~ "ns"
  )
  
  all_pvalues_df$Significant_FDR <- case_when(
    all_pvalues_df$p_value_FDR < 0.001 ~ "***",
    all_pvalues_df$p_value_FDR < 0.01 ~ "**",
    all_pvalues_df$p_value_FDR < 0.05 ~ "*",
    TRUE ~ "ns"
  )
  
  # Reorder columns for clarity
  all_pvalues_df <- all_pvalues_df %>%
    select(Analysis, Pathway, Comparison, ISI, FDR_Group, 
           estimate, SE, df, t_ratio, 
           Main_Effect_p, Interaction_p,
           p_value_uncorrected, Significant_Uncorrected,
           p_value_FDR, Significant_FDR)
  
  # Save FDR-corrected results with Figure_5_6_ prefix
  output_corrected <- paste0(base_path, 'Figure_5_6_All_Stats_FDR_Corrected.csv')
  write.csv(all_pvalues_df, output_corrected, row.names = FALSE)
  cat("\n✓ Saved FDR-corrected p-values to:", output_corrected, "\n\n")
}

################################################################################
# 12. GENERATE SUMMARY FILE
################################################################################

# cat("==============================================================================\n")
# cat("GENERATING SUMMARY FILE\n")
# cat("==============================================================================\n\n")

# summary_df <- bind_rows(summary_data)
# output_summary <- paste0(base_path, 'Figure_5_6_Stats_Summary_R_Model_N.csv')
# write.csv(summary_df, output_summary, row.names = FALSE)
# cat("✓ Saved long-format summary to:", output_summary, "\n")

# # Also generate a wide-format summary for easier manual validation
# wide_summary <- summary_df %>%
#   pivot_wider(
#     names_from = Analysis,
#     values_from = c(N_WT, N_GNB1),
#     values_fill = 0
#   )

# output_summary_wide <- paste0(base_path, 'Figure_5_6_Stats_Summary_R_Model_N_Wide.csv')
# write.csv(wide_summary, output_summary_wide, row.names = FALSE)
# cat("✓ Saved wide-format summary to:", output_summary_wide, "\n\n")

################################################################################
# 13. GENERATE SIGNIFICANCE MARKERS FILE FOR PLOTTING
#
# This creates a CSV that the Python plotting code reads to annotate figures.
# Marker logic:
#   "#"  = Main effect (Genotype) is significant
#   "!"  = Interaction is significant but NO individual ISI survives FDR
#   "*"  = Specific ISI is significant after FDR correction
#          (only checked if interaction was significant)
################################################################################

cat("==============================================================================\n")
cat("GENERATING SIGNIFICANCE MARKERS FOR PLOTTING\n")
cat("==============================================================================\n\n")

markers_list <- list()

if (nrow(all_anova_df) > 0) {
  # Get all unique analysis/pathway/comparison combinations
  unique_combos <- all_anova_df %>%
    select(Analysis, Pathway, Comparison) %>%
    distinct()
  
  for (i in 1:nrow(unique_combos)) {
    combo <- unique_combos[i, ]
    
    # Get the ANOVA rows for this specific combination
    comp_anova <- all_anova_df %>%
      filter(Analysis == combo$Analysis & 
             Pathway == combo$Pathway & 
             Comparison == combo$Comparison)
    
    # Identify which main effect term to check (Genotype or Drug)
    main_effect_term <- if(grepl("WT_vs_GNB1", combo$Comparison)) "Genotype" else "Drug"
    
    # Extract main effect p-value
    main_row <- comp_anova %>% filter(Effect == main_effect_term)
    main_p <- if(nrow(main_row) > 0) main_row$P_Value[1] else NA
    
    # Extract interaction p-value
    inter_row <- comp_anova %>% filter(grepl(":", Effect))
    inter_p <- if(nrow(inter_row) > 0) inter_row$P_Value[1] else NA
    
    # Find which ISIs are significant after FDR correction
    isi_markers <- c()
    if (exists("all_pvalues_df") && nrow(all_pvalues_df) > 0) {
      isi_markers <- all_pvalues_df %>%
        filter(Analysis == combo$Analysis & 
               Pathway == combo$Pathway & 
               Comparison == combo$Comparison &
               p_value_FDR < 0.05) %>%
        pull(ISI)
    }
    
    # Build marker annotations:
    # "#" = main effect significant regardless of interaction
    main_sig <- ifelse(!is.na(main_p) && main_p < 0.05, "#", "")
    
    # "!" = interaction significant but no individual ISI survived FDR
    inter_sig <- ""
    if (!is.na(inter_p) && inter_p < 0.05) {
      if (length(isi_markers) == 0) {
        inter_sig <- "!"
      }
    }
    
    # Store marker row
    markers_list[[length(markers_list) + 1]] <- list(
      Analysis = combo$Analysis,
      Pathway = combo$Pathway,
      Comparison = combo$Comparison,
      Main_Effect_Marker = main_sig,
      Interaction_Marker = inter_sig,
      ISI10_Marker = ifelse("ISI10" %in% isi_markers, "*", ""),
      ISI25_Marker = ifelse("ISI25" %in% isi_markers, "*", ""),
      ISI50_Marker = ifelse("ISI50" %in% isi_markers, "*", ""),
      ISI100_Marker = ifelse("ISI100" %in% isi_markers, "*", ""),
      ISI300_Marker = ifelse("ISI300" %in% isi_markers, "*", ""),
      Main_Effect_p = main_p,
      Interaction_p = inter_p
    )
  }
}

# Save significance markers with Figure_5_6_ prefix
markers_df <- bind_rows(markers_list)
output_markers <- paste0(base_path, 'Figure_5_6_Significance_Markers.csv')
write.csv(markers_df, output_markers, row.names = FALSE)
cat("✓ Saved significance markers to:", output_markers, "\n\n")

# Print a preview of the markers
print(markers_df %>% select(Analysis, Pathway, Comparison, Main_Effect_Marker, Interaction_Marker, ISI300_Marker))

################################################################################
# 14. FINAL SUMMARY
################################################################################

cat("\n==============================================================================\n")
cat("SUMMARY OF RESULTS\n")
cat("==============================================================================\n\n")

if (nrow(all_pvalues_df) > 0) {
  sig_uncorrected <- all_pvalues_df %>% filter(p_value_uncorrected < 0.05)
  sig_fdr <- all_pvalues_df %>% filter(p_value_FDR < 0.05)
  
  cat("Significant before FDR correction (p < 0.05):", nrow(sig_uncorrected), "\n")
  cat("Significant after FDR correction (p < 0.05):", nrow(sig_fdr), "\n\n")
  
  if (nrow(sig_fdr) > 0) {
    cat("Results significant after FDR correction:\n\n")
    print(sig_fdr %>% select(Analysis, Pathway, Comparison, ISI, p_value_FDR, Significant_FDR))
  }
} else {
  cat("No post-hoc tests were run (no significant interaction effects found).\n")
}

cat("\n==============================================================================\n")
cat("OUTPUT FILES GENERATED:\n")
cat("==============================================================================\n")
cat("  1.", output_anova, "\n")
cat("  2.", output_uncorrected, "\n")
cat("  3.", output_corrected, "\n")
cat("  4.", output_summary, "\n")
cat("  5.", output_markers, "\n")
cat("==============================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("==============================================================================\n")
