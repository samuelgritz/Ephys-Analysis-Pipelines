#!/usr/bin/env Rscript
################################################################################
# Age-Stratified Ephys Analysis & LME Modeling (6-9 weeks vs 10-12 weeks)
#
# STATISTICAL MODEL:
#   Linear Mixed Effects (LME) with random intercept for Animal:
#     response ~ Genotype * Age_Group + (1 | Animal)
#   Animal = Date prefix of Cell_ID (YYYYMMDD)
#
# AGE GROUPS:
#   - 6-9 weeks   (Age <= 9 weeks)
#   - 10-12 weeks (Age >= 10 weeks)
#
# OUTPUT:
#   - paper_data/Age_Stratified_R_Stats_Results.csv
################################################################################

suppressPackageStartupMessages({
  library(nlme)
})

cat("\n==============================================================================\n")
cat("AGE-STRATIFIED EPHYS STATISTICAL ANALYSIS (R / LME)\n")
cat("==============================================================================\n\n")

# Determine base path
if (dir.exists("paper_data")) {
  base_dir <- "paper_data"
} else if (dir.exists("../paper_data")) {
  base_dir <- "../paper_data"
} else {
  stop("ERROR: 'paper_data' directory not found.")
}

mdf_path <- ifelse(file.exists("master_df.csv"), "master_df.csv", "../master_df.csv")
if (!file.exists(mdf_path)) {
  stop("ERROR: 'master_df.csv' not found.")
}

mdf <- read.csv(mdf_path, stringsAsFactors = FALSE)
mdf$Cell_ID <- trimws(as.character(mdf$Cell_ID))
mdf$Age_Weeks <- as.numeric(mdf$Mouse.Age..weeks.)
mdf$Age_Group <- ifelse(!is.na(mdf$Age_Weeks) & mdf$Age_Weeks <= 9, "6-9 weeks",
                 ifelse(!is.na(mdf$Age_Weeks) & mdf$Age_Weeks >= 10, "10-12 weeks", NA))

mdf_map <- mdf[, c("Cell_ID", "Age_Group", "Mouse.Age..weeks.")]
mdf_map <- mdf_map[!duplicated(mdf_map$Cell_ID), ]

results_list <- list()

record_lme <- function(fig_label, subpanel, metric_name, pathway, condition, df_sub, val_col) {
  df_sub$Cell_ID  <- trimws(as.character(df_sub$Cell_ID))
  df_sub$Genotype <- ifelse(df_sub$Genotype == "GNB1", "I80T/+", df_sub$Genotype)
  
  if (!"Age_Group" %in% colnames(df_sub)) {
    df_sub <- merge(df_sub, mdf_map, by = "Cell_ID", all.x = TRUE)
  }
  
  df_sub$Animal <- sapply(strsplit(df_sub$Cell_ID, "_"), `[`, 1)
  
  clean <- df_sub[!is.na(df_sub$Age_Group) & !is.na(df_sub[[val_col]]) & df_sub$Genotype %in% c("WT", "I80T/+"), ]
  if (nrow(clean) < 8) return(NULL)
  
  clean$Genotype  <- factor(clean$Genotype, levels = c("WT", "I80T/+"))
  clean$Age_Group <- factor(clean$Age_Group, levels = c("6-9 weeks", "10-12 weeks"))
  
  # Fit 2-Way ANOVA / LME model
  p_geno <- NA; f_geno <- NA
  p_age  <- NA; f_age  <- NA
  p_int  <- NA; f_int  <- NA
  
  tryCatch({
    fit <- lme(as.formula(paste(val_col, "~ Genotype * Age_Group")),
               random = ~ 1 | Animal, data = clean, na.action = na.omit)
    anv <- anova(fit)
    p_geno <- anv["Genotype", "p-value"]; f_geno <- anv["Genotype", "F-value"]
    p_age  <- anv["Age_Group", "p-value"]; f_age  <- anv["Age_Group", "F-value"]
    p_int  <- anv["Genotype:Age_Group", "p-value"]; f_int  <- anv["Genotype:Age_Group", "F-value"]
  }, error = function(e) {
    tryCatch({
      fit_lm <- lm(as.formula(paste(val_col, "~ Genotype * Age_Group")), data = clean)
      anv_lm <- anova(fit_lm)
      p_geno <<- anv_lm["Genotype", "Pr(>F)"]; f_geno <<- anv_lm["Genotype", "F value"]
      p_age  <<- anv_lm["Age_Group", "Pr(>F)"]; f_age  <<- anv_lm["Age_Group", "F value"]
      p_int  <<- anv_lm["Genotype:Age_Group", "Pr(>F)"]; f_int  <<- anv_lm["Genotype:Age_Group", "F value"]
    }, error = function(e2) {})
  })

  # Tukey HSD Post-Hoc Analysis
  clean$Group <- factor(paste0(clean$Genotype, " (", clean$Age_Group, ")"))
  tuk_res <- tryCatch({
    TukeyHSD(aov(as.formula(paste(val_col, "~ Group")), data = clean))
  }, error = function(e) NULL)

  get_tukey_p <- function(g1, g2) {
    if (is.null(tuk_res) || is.null(tuk_res$Group)) return(NA)
    mat <- tuk_res$Group
    rn  <- rownames(mat)
    p1  <- paste0(g1, "-", g2)
    p2  <- paste0(g2, "-", g1)
    if (p1 %in% rn) return(mat[p1, "p adj"])
    if (p2 %in% rn) return(mat[p2, "p adj"])
    return(NA)
  }

  p_tuk_geno_6_9   <- get_tukey_p("WT (6-9 weeks)", "I80T/+ (6-9 weeks)")
  p_tuk_geno_10_12 <- get_tukey_p("WT (10-12 weeks)", "I80T/+ (10-12 weeks)")
  p_tuk_wt_age     <- get_tukey_p("WT (6-9 weeks)", "WT (10-12 weeks)")
  p_tuk_mut_age    <- get_tukey_p("I80T/+ (6-9 weeks)", "I80T/+ (10-12 weeks)")

  # Across-age Mann-Whitney fallback
  wt_6_9   <- clean[clean$Age_Group == "6-9 weeks" & clean$Genotype == "WT", val_col]
  wt_10_12 <- clean[clean$Age_Group == "10-12 weeks" & clean$Genotype == "WT", val_col]
  mut_6_9  <- clean[clean$Age_Group == "6-9 weeks" & clean$Genotype == "I80T/+", val_col]
  mut_10_12<- clean[clean$Age_Group == "10-12 weeks" & clean$Genotype == "I80T/+", val_col]
  
  p_wt_age  <- if (length(wt_6_9) > 0 && length(wt_10_12) > 0) wilcox.test(wt_6_9, wt_10_12, exact = FALSE)$p.value else NA
  p_mut_age <- if (length(mut_6_9) > 0 && length(mut_10_12) > 0) wilcox.test(mut_6_9, mut_10_12, exact = FALSE)$p.value else NA
  
  # Subgroup summary
  for (ag in c("6-9 weeks", "10-12 weeks")) {
    sub_ag <- clean[clean$Age_Group == ag, ]
    wt_vals <- sub_ag[sub_ag$Genotype == "WT", val_col]
    mut_vals <- sub_ag[sub_ag$Genotype == "I80T/+", val_col]
    
    wt_n <- length(wt_vals)
    mut_n <- length(mut_vals)
    
    wt_m <- mean(wt_vals, na.rm = TRUE)
    wt_s <- sd(wt_vals, na.rm = TRUE) / sqrt(wt_n)
    
    mut_m <- mean(mut_vals, na.rm = TRUE)
    mut_s <- sd(mut_vals, na.rm = TRUE) / sqrt(mut_n)
    
    # Wilcoxon / Mann-Whitney post-hoc within age
    p_post <- NA
    u_stat <- NA
    if (wt_n > 0 && mut_n > 0) {
      wt_test <- wilcox.test(wt_vals, mut_vals, exact = FALSE)
      p_post <- wt_test$p.value
      u_stat <- wt_test$statistic
    }
    
    sig_post <- ifelse(is.na(p_post), "ns",
                ifelse(p_post < 0.001, "***",
                ifelse(p_post < 0.01, "**",
                ifelse(p_post < 0.05, "*", "ns"))))
    
    p_tuk_curr <- if (ag == "6-9 weeks") p_tuk_geno_6_9 else p_tuk_geno_10_12

    row_entry <- data.frame(
      Figure = fig_label,
      Subpanel = subpanel,
      Metric = metric_name,
      Pathway = pathway,
      Condition = condition,
      Age_Group = ag,
      WT_Mean = wt_m, WT_SEM = wt_s, WT_N = wt_n,
      I80T_Mean = mut_m, I80T_SEM = mut_s, I80T_N = mut_n,
      ANOVA_F_Genotype = f_geno, ANOVA_P_Genotype = p_geno,
      ANOVA_F_Age = f_age, ANOVA_P_Age = p_age,
      ANOVA_F_Interaction = f_int, ANOVA_P_Interaction = p_int,
      Tukey_P_WithinAge = p_tuk_curr,
      Tukey_P_WT_Age = p_tuk_wt_age,
      Tukey_P_Mutant_Age = p_tuk_mut_age,
      PostHoc_MW_U = u_stat, PostHoc_P_Value = p_post, PostHoc_Sig = sig_post,
      AcrossAge_WT_P_Value = p_wt_age,
      AcrossAge_Mutant_P_Value = p_mut_age,
      stringsAsFactors = FALSE
    )
    results_list[[length(results_list) + 1]] <<- row_entry
  }
}

# --- 1. Intrinsic Properties (Figure 2) ---
cat("Analyzing Intrinsic Properties (Figure 2) …\n")
intr_file <- file.path(base_dir, "Physiology_Analysis", "Intrinsic_properties.csv")
if (file.exists(intr_file)) {
  df_intr <- read.csv(intr_file, stringsAsFactors = FALSE)
  for (col_name in c("Input_Resistance_MOhm", "Vm rest/start (mV)", "Access Resistance (From Whole Cell V-Clamp)", "Voltage_sag")) {
    if (col_name %in% colnames(df_intr)) {
      record_lme("Figure 2", "Somatic Properties", col_name, "N/A", "Control", df_intr, col_name)
    }
  }
}

# --- 2. FI Midpoint (Figure 2) ---
cat("Analyzing F-I Midpoint (Figure 2) …\n")
fi_file <- file.path(base_dir, "Firing_Rate", "Sigmoid_Fit_Params.csv")
if (file.exists(fi_file)) {
  df_fi <- read.csv(fi_file, stringsAsFactors = FALSE)
  val_c <- ifelse("Midpoint" %in% colnames(df_fi), "Midpoint", "FI_Midpoint")
  if (val_c %in% colnames(df_fi)) {
    record_lme("Figure 2", "Somatic Excitability", "F-I Curve Midpoint (pA)", "N/A", "Control", df_fi, val_c)
  }
}

# --- 3. Unitary EPSP & Inhibition (Figure 4) ---
cat("Analyzing Unitary E:I (Figure 4) …\n")
ei_file <- file.path(base_dir, "E_I_data", "E_I_amplitudes.csv")
if (file.exists(ei_file)) {
  df_ei <- read.csv(ei_file, stringsAsFactors = FALSE)
  for (pw in c("Perforant", "Schaffer", "Basal_Stratum_Oriens")) {
    sub_pw <- df_ei[df_ei$Pathway == pw & df_ei$ISI == 300, ]
    if (nrow(sub_pw) > 0) {
      if ("Gabazine_Amplitude" %in% colnames(sub_pw)) {
        record_lme("Figure 4", "Unitary EPSP", "Gabazine EPSP Amplitude (mV)", pw, "Gabazine", sub_pw, "Gabazine_Amplitude")
      }
      if ("Estimated_Inhibition_Amplitude" %in% colnames(sub_pw)) {
        record_lme("Figure 4", "Unitary Inhibition", "GABAA Inhibition Amplitude (mV)", pw, "Estimated Inhibition", sub_pw, "Estimated_Inhibition_Amplitude")
      }
      if ("GABAB_Area" %in% colnames(sub_pw)) {
        record_lme("Figure 4", "GABAB Area", "GABAB Area (mV·s)", pw, "GABAB", sub_pw, "GABAB_Area")
      }
      if ("E_I_Imbalance" %in% colnames(sub_pw)) {
        record_lme("Figure 4", "E/I Imbalance Index", "E/I Imbalance Index (ISI 300ms)", pw, "Control", sub_pw, "E_I_Imbalance")
      }
    }
  }
}

# --- 4. Dendritic Plateau Area (Figure 7) ---
cat("Analyzing Dendritic Plateau Area (Figure 7) …\n")
plat_file <- file.path(base_dir, "Plateau_data", "Plateau_data.csv")
if (file.exists(plat_file)) {
  df_plat <- read.csv(plat_file, stringsAsFactors = FALSE)
  df_gab <- df_plat[grepl("Gabazine", df_plat$Condition, ignore.case = TRUE), ]
  if (nrow(df_gab) > 0 && "Plateau_Area" %in% colnames(df_gab)) {
    record_lme("Figure 7", "Dendritic Excitability", "Plateau Area (mV·s)", "Both Pathways", "Gabazine", df_gab, "Plateau_Area")
  }
}

# Combine and Export Results
if (length(results_list) > 0) {
  out_df <- do.call(rbind, results_list)
  out_csv <- file.path(base_dir, "Age_Stratified_R_Stats_Results.csv")
  write.csv(out_df, out_csv, row.names = FALSE)
  cat("\n✓ Exported Age-Stratified R Stats to:", out_csv, "(", nrow(out_df), "rows)\n")
} else {
  cat("\n⚠ No results generated.\n")
}

cat("\nDone.\n")
