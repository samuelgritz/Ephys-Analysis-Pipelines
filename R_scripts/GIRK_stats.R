# GIRK Channel Statistics Script
# Runs LME analysis for ML297 and ETX effects
# Updated to use emmeans with Tukey adjustment (consistent with Figure 4 and user request)

library(nlme)
library(car)
library(dplyr)
library(emmeans)
library(readr)

# ==============================================================================
# 1. LOAD AND PREPARE DATA
# ==============================================================================

# Determine data root to allow running from different PWDs
if (file.exists("paper_data")) {
  data_root <- "paper_data"
} else if (file.exists("../paper_data")) {
  data_root <- "../paper_data"
} else {
  # Fallback to current dir if csv exists here
  data_root <- "." 
}

# Try to find the file
file_path <- file.path(data_root, "GIRK_df.csv")
if (!file.exists(file_path)) {
   # Try explicit list from previous listing
   if (file.exists("GIRK_df.csv")) file_path <- "GIRK_df.csv"
}

if (!file.exists(file_path)) {
  stop("Could not find GIRK_df.csv")
}

raw_df <- read.csv(file_path)

custom_levels <- c(
  "WT.ML297.Before",
  "WT.ML297.After",
  "GNB1.ML297.Before",
  "GNB1.ML297.After",
  "WT.ETX.Before",
  "WT.ETX.After",
  "GNB1.ETX.Before",
  "GNB1.ETX.After"
)

conjugated_df <- raw_df %>%
  mutate(
    Group = factor(paste(Genotype, Drug, Condition, sep = "_")),
    Subject = factor(Subject),
    Genotype_Drug_Condition = interaction(Genotype, Drug, Condition),
    Genotype_Drug_Condition = factor(Genotype_Drug_Condition, levels = custom_levels),
    # Separate factors for emmeans
    Genotype = factor(Genotype),
    Drug = factor(Drug),
    Condition = factor(Condition, levels = c("Before", "After"))
  )

# ==============================================================================
# 2. WITHIN-GROUP CONTRASTS (Before vs After)
# ==============================================================================

# Model LME
mod_lme <- lme(
  Plateau_Area ~ Genotype * Drug * Condition,
  random = ~1 | Subject,
  data = conjugated_df
)

anova_results <- Anova(mod_lme, type = 3)
print("=== ANOVA Results (Condition Effect) ===")
print(anova_results)

# Post-hoc: Compare After vs Before within each Genotype*Drug combination
# Uses Tukey adjustment within each family (which is size 1 here, so uncorrected P, 
# but consistent with "Tukey within condition" strategy in Figure 4)
emm_means <- emmeans(mod_lme, pairwise ~ Condition | Genotype * Drug)

# We specifically want After - Before.
# The default pairwise usually gives Before - After. 
# We reverse to get After - Before (positive = increase).
contrasts_means <- test(emm_means$contrasts, adjust = "tukey")
contrasts_means_df <- as.data.frame(contrasts_means)

# Add significance stars
contrasts_means_df <- contrasts_means_df %>%
  mutate(
    significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

print("=== Within-Group Contrasts (After vs Before) ===")
print(contrasts_means_df)

# ==============================================================================
# 3. BETWEEN-GROUP SLOPE COMPARISONS
# ==============================================================================

# Prepare numeric condition for slope analysis
# Slope = (After - Before) / 1 unit
conjugated_df2 <- raw_df %>%
  mutate(
    Subject = factor(Subject),
    Condition = factor(Condition, levels = c("Before", "After")),
    Condition_num = ifelse(Condition == "Before", 0, 1),
    Genotype = factor(Genotype),
    Drug = factor(Drug)
  )

mod_lme2 <- lme(
  Plateau_Area ~ Genotype * Drug * Condition_num,
  random = ~1 | Subject,
  data = conjugated_df2
)

# Compare Slopes (Condition_num trend) between Genotypes, within each Drug
# emtrends computes the slope for each Genotype*Drug combination
# pairwise ~ Genotype | Drug compares WT vs GNB1 within each Drug.
emm_slope <- emtrends(mod_lme2, pairwise ~ Genotype | Drug, var = "Condition_num")
contrasts_slope <- test(emm_slope$contrasts, adjust = "tukey")

contrasts_slope_df <- as.data.frame(contrasts_slope)
contrasts_slope_df <- contrasts_slope_df %>%
  mutate(
    significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

print("=== Between-Group Slope Comparisons (WT vs GNB1) ===")
print(contrasts_slope_df)

# ==============================================================================
# 4. SAVE RESULTS
# ==============================================================================

# Combine results for single output
means_out <- contrasts_means_df %>%
  mutate(Type = "Within_Group_Change") %>%
  dplyr::select(Type, Genotype, Drug, contrast, estimate, SE, df, t.ratio, p.value, significance)

slopes_out <- contrasts_slope_df %>%
  mutate(Type = "Between_Group_Slope_Diff", contrast = paste(contrast, "slope")) %>%
  dplyr::select(Type, Drug, contrast, estimate, SE, df, t.ratio, p.value, significance) %>%
  mutate(Genotype = "Combined") # Placeholder

combined_out <- bind_rows(means_out, slopes_out)

write.csv(combined_out, "GIRK_PostHoc_Stats.csv", row.names = FALSE)
print(paste("Saved results to GIRK_PostHoc_Stats.csv"))
