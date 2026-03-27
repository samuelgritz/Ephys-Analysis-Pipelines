#!/usr/bin/env Rscript
################################################################################
# Figure 4 - All Stats with FDR Correction
# 
# This script performs all statistical analyses for Figure 4:
# - EPSP Amplitudes (per pathway, per condition)
# - Gabazine Supralinearity
# - E:I Imbalance
#
# OUTPUTS:
# - Figure_4_All_Stats_ANOVA.csv          - All ANOVA results
# - Figure_4_All_Stats_Uncorrected.csv    - All uncorrected post-hoc p-values  
# - Figure_4_All_Stats_FDR_Corrected.csv  - FDR-corrected p-values by group
# - Figure_4_Stats_Summary.csv            - Summary with N per pathway/genotype
# - Figure_4_Significance_Markers.csv     - Markers for plotting (# main, * post-hoc)
################################################################################

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)

################################################################################
# 1. LOAD AND PREPARE DATA
################################################################################

cat("\n==============================================================================\n")
cat("FIGURE 4: ALL STATS WITH FDR CORRECTION\n")
cat("==============================================================================\n\n")

# Determine base path for data
if (dir.exists("paper_data")) {
  base_path <- "paper_data/E_I_data/"
} else if (dir.exists("../paper_data")) {
  base_path <- "../paper_data/E_I_data/"
} else {
  stop("ERROR: 'paper_data' directory not found. Please run from project root or R_scripts directory.")
}

# Load E_I EPSP amplitude data
file_name <- paste0(base_path, 'E_I_EPSP_amplitudes_R_format.csv')

if (!file.exists(file_name)) {
  stop("ERROR: Data file not found: ", file_name, "\n",
       "Please run Analyze_and_Export_E_I_data.py first to generate data.")
}

E_I_experiment <- read.csv(file_name)
cat("✓ Loaded EPSP amplitude data from:", file_name, "\n")
cat("  Rows:", nrow(E_I_experiment), "\n\n")

# Prepare data
ISI_cols <- c("ISI10", "ISI25", "ISI50", "ISI100", "ISI300")
E_I_experiment_clean <- E_I_experiment %>%
  filter(complete.cases(!!!syms(ISI_cols)))

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
    ISI_Time = factor(ISI_Time, levels = ISI_cols)
  )

cat("  Converted to long format:", nrow(E_I_experiment_long), "rows\n")
cat("  Subjects (cells):", length(unique(E_I_experiment_long$Subject)), "\n\n")

# Load E_I data for Supralinearity and E:I Imbalance
file_name_ei <- paste0(base_path, 'E_I_amplitudes.csv')

if (!file.exists(file_name_ei)) {
  stop("ERROR: Data file not found: ", file_name_ei, "\n",
       "Please run Analyze_and_Export_E_I_data.py first to generate data.")
}

E_I_data <- read.csv(file_name_ei)
cat("✓ Loaded Supralinearity and E:I Imbalance data from:", file_name_ei, "\n")
cat("  Rows:", nrow(E_I_data), "\n\n")

E_I_clean <- E_I_data %>%
  mutate(
    Subject = as.factor(Cell_ID),
    Genotype = as.factor(Genotype),
    Pathway = as.factor(Pathway),
    ISI_Time = as.factor(paste0("ISI", ISI))
  )

################################################################################
# 2. INITIALIZE STORAGE LISTS
################################################################################

all_anova_results <- list()
all_uncorrected_pvalues <- list()
summary_data <- list()

# Pathway name mapping
pathway_names <- c("1" = "Perforant", "2" = "Schaffer", "3" = "Basal_Stratum_Oriens")

################################################################################
# 3. HELPER FUNCTION: Extract ANOVA and Post-hoc results
################################################################################

run_lmer_analysis <- function(data, formula_str, analysis_name, pathway_name, 
                               comparison_name, fdr_group, effect_to_test = "interaction") {
  
  # Fit model
  model <- tryCatch({
    lmer(as.formula(formula_str), data = data, REML = TRUE)
  }, error = function(e) {
    cat("    Error fitting model:", conditionMessage(e), "\n")
    return(NULL)
  })
  
  if (is.null(model)) return(list(anova = NULL, posthoc = NULL))
  
  # Get ANOVA
  anova_result <- anova(model)
  anova_df <- as.data.frame(anova_result)
  anova_df$Effect <- rownames(anova_df)
  colnames(anova_df)[colnames(anova_df) == "Pr(>F)"] <- "P_Value"
  anova_df$Analysis <- analysis_name
  anova_df$Pathway <- pathway_name
  anova_df$Comparison <- comparison_name
  
  # Calculate Means and SEMs for the comparison groups
  group_col <- if (grepl("Genotype", formula_str)) "Genotype" else "Drug"
  
  # Group-wise summary
  summary_stats <- data %>%
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
  
  # Add summary stats to every row of the ANOVA dataframe
  anova_df <- cbind(anova_df, summary_stats)
  
  anova_df$Significant <- ifelse(anova_df$P_Value < 0.05, "*", "ns")
  
  # Determine which term to test for post-hoc
  if (effect_to_test == "interaction") {
    # Look for interaction term (contains ":")
    interaction_rows <- grep(":", anova_df$Effect)
    if (length(interaction_rows) > 0) {
      p_val <- anova_df$P_Value[interaction_rows[1]]
      main_p <- anova_df$P_Value[1]  # First main effect
    } else {
      p_val <- NA
      main_p <- anova_df$P_Value[1]
    }
  } else {
    p_val <- anova_df$P_Value[1]  # Main effect
    main_p <- p_val
  }
  
  # Post-hoc if interaction significant
  posthoc_list <- list()
  
  if (!is.na(p_val) && p_val < 0.05) {
    # Determine contrast type based on formula
    if (grepl("Genotype", formula_str) && grepl("ISI_Time", formula_str)) {
      emm <- emmeans(model, pairwise ~ Genotype | ISI_Time)
    } else if (grepl("Drug", formula_str) && grepl("ISI_Time", formula_str)) {
      emm <- emmeans(model, pairwise ~ Drug | ISI_Time)
    } else {
      emm <- NULL
    }
    
    if (!is.null(emm)) {
      contrasts <- summary(emm$contrasts, adjust = "none")
      
      for (i in 1:nrow(contrasts)) {
        posthoc_list[[length(posthoc_list) + 1]] <- list(
          Analysis = analysis_name,
          Pathway = pathway_name,
          Comparison = comparison_name,
          ISI = as.character(contrasts$ISI_Time[i]),
          estimate = contrasts$estimate[i],
          SE = contrasts$SE[i],
          df = contrasts$df[i],
          t_ratio = contrasts$t.ratio[i],
          p_value_uncorrected = contrasts$p.value[i],
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
# 4. EPSP AMPLITUDE ANALYSES - ALL PATHWAYS
################################################################################

cat("==============================================================================\n")
cat("RUNNING EPSP AMPLITUDE ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_code in c("1", "2", "3")) {
  pathway_name <- pathway_names[pathway_code]
  cat("\n--- ", pathway_name, " PATHWAY ---\n\n", sep = "")
  
  # Count cells for summary
  pathway_data <- E_I_experiment_long %>% filter(Pathway == pathway_code)
  n_wt <- length(unique(pathway_data$Subject[pathway_data$Genotype == "WT"]))
  n_gnb1 <- length(unique(pathway_data$Subject[pathway_data$Genotype == "GNB1"]))
  
  summary_data[[paste(pathway_name, "EPSP_Amplitude", sep = "_")]] <- list(
    Pathway = pathway_name,
    Analysis = "EPSP_Amplitude",
    N_WT = n_wt,
    N_GNB1 = n_gnb1
  )
  
  # 1. WT vs GNB1 (Gabazine only)
  gabazine_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Drug == "Gabazine")
  result_gab <- run_lmer_analysis(
    data = gabazine_data,
    formula_str = "EPSP_Amplitude ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1_Gabazine",
    fdr_group = paste0(substr(pathway_name, 1, 2), "_EPSP")
  )
  
  if (!is.null(result_gab$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_gab$anova
    cat("  WT vs GNB1 (Gabazine): Interaction p =", 
        ifelse(is.na(result_gab$interaction_p), "NA", round(result_gab$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_gab$posthoc)
  
  # 2. WT vs GNB1 (Control only)
  control_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Drug == "Control")
  result_ctrl <- run_lmer_analysis(
    data = control_data,
    formula_str = "EPSP_Amplitude ~ Genotype + Genotype * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_vs_GNB1_Control",
    fdr_group = paste0(substr(pathway_name, 1, 2), "_EPSP")
  )
  
  if (!is.null(result_ctrl$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_ctrl$anova
    cat("  WT vs GNB1 (Control): Interaction p =", 
        ifelse(is.na(result_ctrl$interaction_p), "NA", round(result_ctrl$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_ctrl$posthoc)
  
  # 3. WT: Control vs Gabazine
  wt_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Genotype == "WT")
  result_wt <- run_lmer_analysis(
    data = wt_data,
    formula_str = "EPSP_Amplitude ~ Drug + Drug * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "WT_Control_vs_Gabazine",
    fdr_group = paste0(substr(pathway_name, 1, 2), "_EPSP")
  )
  
  if (!is.null(result_wt$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result_wt$anova
    cat("  WT Control vs Gabazine: Interaction p =", 
        ifelse(is.na(result_wt$interaction_p), "NA", round(result_wt$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result_wt$posthoc)
  
  # 4. GNB1: Control vs Gabazine
  gnb1_data <- E_I_experiment_long %>% filter(Pathway == pathway_code & Genotype == "GNB1")
  result_gnb1 <- run_lmer_analysis(
    data = gnb1_data,
    formula_str = "EPSP_Amplitude ~ Drug + Drug * ISI_Time + (1 | Subject)",
    analysis_name = "EPSP_Amplitude",
    pathway_name = pathway_name,
    comparison_name = "GNB1_Control_vs_Gabazine",
    fdr_group = paste0(substr(pathway_name, 1, 2), "_EPSP")
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
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING GABAZINE SUPRALINEARITY ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
  cat("--- ", pathway_name, " ---\n", sep = "")
  
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
    fdr_group = paste0("Supra_", substr(pathway_name, 1, 2))
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
################################################################################

cat("\n==============================================================================\n")
cat("RUNNING E:I IMBALANCE ANALYSES\n")
cat("==============================================================================\n\n")

for (pathway_name in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
  cat("--- ", pathway_name, " ---\n", sep = "")
  
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
    fdr_group = paste0("EI_", substr(pathway_name, 1, 2))
  )
  
  if (!is.null(result$anova)) {
    all_anova_results[[length(all_anova_results) + 1]] <- result$anova
    cat("  Genotype:ISI_Time p =", 
        ifelse(is.na(result$interaction_p), "NA", round(result$interaction_p, 4)), "\n")
  }
  all_uncorrected_pvalues <- c(all_uncorrected_pvalues, result$posthoc)
}

################################################################################
# 7. COMPILE AND SAVE ANOVA RESULTS
################################################################################

cat("\n==============================================================================\n")
cat("SAVING ANOVA RESULTS\n")
cat("==============================================================================\n\n")

# Combine all ANOVA results
all_anova_df <- bind_rows(all_anova_results) %>%
  select(Analysis, Pathway, Comparison, Effect, 
         starts_with("Mean_"), starts_with("SEM_"),
         `Sum Sq`, `Mean Sq`, NumDF, DenDF, `F value`, P_Value, Significant) %>%
  arrange(Analysis, Pathway, Comparison, Effect)

output_anova <- paste0(base_path, 'Figure_4_All_Stats_ANOVA.csv')
write.csv(all_anova_df, output_anova, row.names = FALSE)
cat("✓ Saved ANOVA results to:", output_anova, "\n")
cat("  Total ANOVA tests:", nrow(all_anova_df), "\n\n")

################################################################################
# 8. COMPILE AND SAVE UNCORRECTED P-VALUES
################################################################################

cat("==============================================================================\n")
cat("COMPILING UNCORRECTED P-VALUES\n")
cat("==============================================================================\n\n")

# Convert list to dataframe
all_pvalues_df <- bind_rows(all_uncorrected_pvalues)

cat("Total uncorrected p-values collected:", nrow(all_pvalues_df), "\n")
if (nrow(all_pvalues_df) > 0) {
  cat("By FDR Group:\n")
  print(table(all_pvalues_df$FDR_Group))
}
cat("\n")

# Save uncorrected p-values
output_uncorrected <- paste0(base_path, 'Figure_4_All_Stats_Uncorrected.csv')
write.csv(all_pvalues_df, output_uncorrected, row.names = FALSE)
cat("✓ Saved uncorrected p-values to:", output_uncorrected, "\n\n")

################################################################################
# 9. APPLY FDR CORRECTION BY GROUPS
################################################################################

cat("==============================================================================\n")
cat("APPLYING FDR CORRECTION BY GROUPS\n")
cat("==============================================================================\n\n")

if (nrow(all_pvalues_df) > 0) {
  # Apply FDR correction separately for each group
  all_pvalues_df$p_value_FDR <- NA
  
  groups <- unique(all_pvalues_df$FDR_Group)
  
  for (grp in groups) {
    idx <- which(all_pvalues_df$FDR_Group == grp)
    if (length(idx) > 0) {
      all_pvalues_df$p_value_FDR[idx] <- p.adjust(all_pvalues_df$p_value_uncorrected[idx], method = "fdr")
      cat("✓ Applied FDR correction to group:", grp, "(", length(idx), "tests )\n")
    }
  }
  
  # Add significance indicators
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
  
  # Reorder columns
  all_pvalues_df <- all_pvalues_df %>%
    select(Analysis, Pathway, Comparison, ISI, FDR_Group, 
           estimate, SE, df, t_ratio, 
           Main_Effect_p, Interaction_p,
           p_value_uncorrected, Significant_Uncorrected,
           p_value_FDR, Significant_FDR)
  
  # Save FDR-corrected results
  output_corrected <- paste0(base_path, 'Figure_4_All_Stats_FDR_Corrected.csv')
  write.csv(all_pvalues_df, output_corrected, row.names = FALSE)
  cat("\n✓ Saved FDR-corrected p-values to:", output_corrected, "\n\n")
}

################################################################################
# 10. GENERATE SUMMARY FILE
################################################################################

cat("==============================================================================\n")
cat("GENERATING SUMMARY FILE\n")
cat("==============================================================================\n\n")

summary_df <- bind_rows(summary_data)
output_summary <- paste0(base_path, 'Figure_4_Stats_Summary.csv')
write.csv(summary_df, output_summary, row.names = FALSE)
cat("✓ Saved summary to:", output_summary, "\n\n")

################################################################################
# 11. GENERATE SIGNIFICANCE MARKERS FILE FOR PLOTTING
################################################################################

cat("==============================================================================\n")
cat("GENERATING SIGNIFICANCE MARKERS FOR PLOTTING\n")
cat("==============================================================================\n\n")

# Create markers dataframe
# Main effect significant = "#"
# Interaction significant + post-hoc FDR significant = "*" at specific ISI

markers_list <- list()

if (nrow(all_anova_df) > 0) {
  # Get unique analysis/pathway/comparison combinations from all ANOVA results
  unique_combos <- all_anova_df %>%
    select(Analysis, Pathway, Comparison) %>%
    distinct()
  
  for (i in 1:nrow(unique_combos)) {
    combo <- unique_combos[i, ]
    
    # Get all ANOVA rows for this specific comparison
    comp_anova <- all_anova_df %>%
      filter(Analysis == combo$Analysis & 
             Pathway == combo$Pathway & 
             Comparison == combo$Comparison)
    
    # Identify the relevant main effect term (Genotype or Drug)
    main_effect_term <- if(grepl("WT_vs_GNB1", combo$Comparison)) "Genotype" else "Drug"
    
    # Get Main Effect p-value
    main_row <- comp_anova %>% filter(Effect == main_effect_term)
    main_p <- if(nrow(main_row) > 0) main_row$P_Value[1] else NA
    
    # Get Interaction p-value
    inter_row <- comp_anova %>% filter(grepl(":", Effect))
    inter_p <- if(nrow(inter_row) > 0) inter_row$P_Value[1] else NA
    
    # Main effect significant = "#"
    main_sig <- ifelse(!is.na(main_p) && main_p < 0.05, "#", "")
    
    # Check each ISI for post-hoc significance from the FDR-corrected results
    isi_markers <- c()
    if (exists("all_pvalues_df") && nrow(all_pvalues_df) > 0) {
      isi_markers <- all_pvalues_df %>%
        filter(Analysis == combo$Analysis & 
               Pathway == combo$Pathway & 
               Comparison == combo$Comparison &
               p_value_FDR < 0.05) %>%
        pull(ISI)
    }
    
    markers_list[[length(markers_list) + 1]] <- list(
      Analysis = combo$Analysis,
      Pathway = combo$Pathway,
      Comparison = combo$Comparison,
      Main_Effect_Marker = main_sig,
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

markers_df <- bind_rows(markers_list)
output_markers <- paste0(base_path, 'Figure_4_Significance_Markers.csv')
write.csv(markers_df, output_markers, row.names = FALSE)
cat("✓ Saved significance markers to:", output_markers, "\n\n")

print(markers_df %>% select(Analysis, Pathway, Comparison, Main_Effect_Marker, ISI10_Marker, ISI25_Marker, ISI50_Marker, ISI100_Marker, ISI300_Marker))

################################################################################
# 12. FINAL SUMMARY
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
