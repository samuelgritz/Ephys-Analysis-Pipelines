#!/usr/bin/env Rscript
################################################################################
# Continuous Age Regressions & ANCOVA (Mouse Age in Weeks vs Ephys Properties)
#
# MODELS:
#   1. Linear Model (LM): response ~ Age_Weeks * Genotype
#   2. Linear Mixed Effects (LME): response ~ Age_Weeks * Genotype + (1 | Animal)
#
# METRICS ANALYZED:
#   - Somatic Input Resistance (Input_Resistance_MOhm)
#   - F-I Curve Midpoint (Midpoint)
#   - Perforant Path GABAB Area (GABAB_Area)
#   - Dendritic Plateau Area (Plateau_Area)
#
# OUTPUT:
#   - paper_data/Age_Continuous_R_Stats_Results.csv
################################################################################

suppressPackageStartupMessages({
  library(nlme)
})

cat("\n==============================================================================\n")
cat("CONTINUOUS AGE REGRESSION & ANCOVA STATISTICAL ANALYSIS (R)\n")
cat("==============================================================================\n\n")

base_dir <- ifelse(dir.exists("paper_data"), "paper_data", "../paper_data")
mdf_path <- ifelse(file.exists("master_df.csv"), "master_df.csv", "../master_df.csv")

if (!file.exists(mdf_path)) {
  stop("ERROR: 'master_df.csv' not found.")
}

mdf <- read.csv(mdf_path, stringsAsFactors = FALSE)
mdf$Cell_ID <- trimws(as.character(mdf$Cell_ID))
mdf$Age_Weeks <- as.numeric(mdf$Mouse.Age..weeks.)

mdf_map <- mdf[, c("Cell_ID", "Age_Weeks")]
mdf_map <- mdf_map[!duplicated(mdf_map$Cell_ID), ]

results_list <- list()

record_continuous_stats <- function(fig_label, subpanel, metric_name, pathway, condition, df_sub, val_col) {
  df_sub$Cell_ID  <- trimws(as.character(df_sub$Cell_ID))
  df_sub$Genotype <- ifelse(df_sub$Genotype == "GNB1", "I80T/+", df_sub$Genotype)
  
  if (!"Age_Weeks" %in% colnames(df_sub)) {
    df_sub <- merge(df_sub, mdf_map, by = "Cell_ID", all.x = TRUE)
  }
  
  df_sub$Animal <- sapply(strsplit(df_sub$Cell_ID, "_"), `[`, 1)
  
  clean <- df_sub[!is.na(df_sub$Age_Weeks) & !is.na(df_sub[[val_col]]) & df_sub$Genotype %in% c("WT", "I80T/+"), ]
  if (nrow(clean) < 8) return(NULL)
  
  clean$Genotype <- factor(clean$Genotype, levels = c("WT", "I80T/+"))
  
  # WT & Mutant individual linear regressions
  wt_clean  <- clean[clean$Genotype == "WT", ]
  mut_clean <- clean[clean$Genotype == "I80T/+", ]
  
  fit_wt  <- lm(as.formula(paste(val_col, "~ Age_Weeks")), data = wt_clean)
  fit_mut <- lm(as.formula(paste(val_col, "~ Age_Weeks")), data = mut_clean)
  
  s_wt  <- summary(fit_wt)
  s_mut <- summary(fit_mut)
  
  wt_slope  <- coef(fit_wt)["Age_Weeks"]
  wt_r2     <- s_wt$r.squared
  wt_p      <- coef(s_wt)["Age_Weeks", "Pr(>|t|)"]
  
  mut_slope <- coef(fit_mut)["Age_Weeks"]
  mut_r2    <- s_mut$r.squared
  mut_p     <- coef(s_mut)["Age_Weeks", "Pr(>|t|)"]
  
  # Full Interaction ANCOVA Model
  fit_ancova <- lm(as.formula(paste(val_col, "~ Age_Weeks * Genotype")), data = clean)
  anv_ancova <- anova(fit_ancova)
  
  p_geno_ancova <- anv_ancova["Genotype", "Pr(>F)"]
  p_age_ancova  <- anv_ancova["Age_Weeks", "Pr(>F)"]
  p_int_ancova  <- anv_ancova["Age_Weeks:Genotype", "Pr(>F)"]
  f_int_ancova  <- anv_ancova["Age_Weeks:Genotype", "F value"]
  
  row_entry <- data.frame(
    Figure = fig_label,
    Subpanel = subpanel,
    Metric = metric_name,
    Pathway = pathway,
    Condition = condition,
    WT_N = nrow(wt_clean),
    WT_Slope = wt_slope,
    WT_R2 = wt_r2,
    WT_P_Value = wt_p,
    I80T_N = nrow(mut_clean),
    I80T_Slope = mut_slope,
    I80T_R2 = mut_r2,
    I80T_P_Value = mut_p,
    ANCOVA_P_Genotype = p_geno_ancova,
    ANCOVA_P_Age = p_age_ancova,
    ANCOVA_F_Interaction = f_int_ancova,
    ANCOVA_P_Interaction = p_int_ancova,
    stringsAsFactors = FALSE
  )
  results_list[[length(results_list) + 1]] <<- row_entry
}

# --- 1. Intrinsic Properties ---
cat("Analyzing Continuous Age Regressions for Somatic Input Resistance …\n")
intr_file <- file.path(base_dir, "Physiology_Analysis", "Intrinsic_properties.csv")
if (file.exists(intr_file)) {
  df_intr <- read.csv(intr_file, stringsAsFactors = FALSE)
  record_continuous_stats("Figure 2", "Somatic Properties", "Input_Resistance_MOhm", "N/A", "Control", df_intr, "Input_Resistance_MOhm")
}

# --- 2. FI Midpoint ---
cat("Analyzing Continuous Age Regressions for F-I Midpoint …\n")
fi_file <- file.path(base_dir, "Firing_Rate", "Sigmoid_Fit_Params.csv")
if (file.exists(fi_file)) {
  df_fi <- read.csv(fi_file, stringsAsFactors = FALSE)
  val_c <- ifelse("Midpoint" %in% colnames(df_fi), "Midpoint", "FI_Midpoint")
  record_continuous_stats("Figure 2", "Somatic Excitability", "F-I Curve Midpoint (pA)", "N/A", "Control", df_fi, val_c)
}

# --- 3. Perforant GABAB Area ---
cat("Analyzing Continuous Age Regressions for Perforant GABAB Area …\n")
ei_file <- file.path(base_dir, "E_I_data", "E_I_amplitudes.csv")
if (file.exists(ei_file)) {
  df_ei <- read.csv(ei_file, stringsAsFactors = FALSE)
  sub_pp <- df_ei[df_ei["Pathway"] == "Perforant" & df_ei["ISI"] == 300, ]
  record_continuous_stats("Figure 4", "GABAB Area", "GABAB Area (mV·s)", "Perforant", "GABAB", sub_pp, "GABAB_Area")
}

# --- 4. Dendritic Plateau Area ---
cat("Analyzing Continuous Age Regressions for Dendritic Plateau Area …\n")
plat_file <- file.path(base_dir, "Plateau_data", "Plateau_data.csv")
if (file.exists(plat_file)) {
  df_plat <- read.csv(plat_file, stringsAsFactors = FALSE)
  df_gab <- df_plat[grepl("Gabazine", df_plat$Condition, ignore.case = TRUE), ]
  record_continuous_stats("Figure 7", "Dendritic Excitability", "Plateau Area (mV·s)", "Both Pathways", "Gabazine", df_gab, "Plateau_Area")
}

if (length(results_list) > 0) {
  out_df <- do.call(rbind, results_list)
  out_csv <- file.path(base_dir, "Age_Continuous_R_Stats_Results.csv")
  write.csv(out_df, out_csv, row.names = FALSE)
  cat("\n✓ Exported Continuous Age R Stats to:", out_csv, "(", nrow(out_df), "rows)\n")
} else {
  cat("\n⚠ No results generated.\n")
}

cat("\nDone.\n")
