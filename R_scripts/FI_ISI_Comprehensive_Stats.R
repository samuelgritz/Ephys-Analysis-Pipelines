# Comprehensive F-I and ISI Statistics
# Includes: F-I curve 2-way ANOVA, F-I slope comparison, ISI adaptation 2-way ANOVA
library(tidyverse)

# Significance helper (used throughout)
sig_func <- function(p) {
  if (is.na(p)) "N/A"
  else if (p < 0.001) "***"
  else if (p < 0.01) "**"
  else if (p < 0.05) "*"
  else "ns"
}

# ============================================================================
# PART 1: F-I SLOPE COMPARISON (commented out - now using midpoint)
# ============================================================================
cat("\n================================================================================\n")
cat("PART 1: F-I SLOPE COMPARISON\n")
cat("================================================================================\n")

# Load data
# Load data with flexible path handling
data_file <- "paper_data/Firing_Rate/Firing_Rates_plotting_format.csv"
if (!file.exists(data_file)) {
  data_file <- "../paper_data/Firing_Rate/Firing_Rates_plotting_format.csv"
}

if (!file.exists(data_file)) {
  stop("Could not find data file in 'paper_data' or '../paper_data'")
}

fi_data <- read.csv(data_file)

# # Extract F-I slopes
# slopes <- fi_data %>%
#   select(Cell_ID, Genotype, FI_Slope) %>%
#   filter(!is.na(FI_Slope))

# cat("\nN per genotype for F-I slope:\n")
# print(table(slopes$Genotype))

# # Normality test
# wt_slopes <- slopes %>% filter(Genotype == "WT") %>% pull(FI_Slope)
# gnb1_slopes <- slopes %>% filter(Genotype == "GNB1") %>% pull(FI_Slope)

# shapiro_wt <- shapiro.test(wt_slopes)
# shapiro_gnb1 <- shapiro.test(gnb1_slopes)

# cat("\nNormality tests:\n")
# cat(sprintf("WT: W = %.4f, p = %.4f\n", shapiro_wt$statistic, shapiro_wt$p.value))
# cat(sprintf("GNB1: W = %.4f, p = %.4f\n", shapiro_gnb1$statistic, shapiro_gnb1$p.value))

# Use Mann-Whitney (consistent with other physiology stats)
# mw_test <- wilcox.test(FI_Slope ~ Genotype, data = slopes)

# sig_func <- function(p) {
#   if (is.na(p)) "N/A"
#   else if (p < 0.001) "***"
#   else if (p < 0.01) "**"
#   else if (p < 0.05) "*"
#   else "ns"
# }

# slope_result <- data.frame(
#   Comparison = "F-I Slope: WT vs GNB1",
#   Test = "Mann-Whitney U",
#   Statistic = mw_test$statistic,
#   p_value = mw_test$p.value,
#   significance = sig_func(mw_test$p.value)
# )

# cat("\nF-I Slope Result:\n")
# print(slope_result)


# ============================================================================
# PART 2: F-I CURVE 2-WAY REPEATED MEASURES ANOVA
# ============================================================================
cat("\n================================================================================\n")
cat("PART 2: F-I CURVE 2-WAY REPEATED MEASURES ANOVA\n")
cat("================================================================================\n")

# Reshape data for ANOVA
library(jsonlite)

fi_long <- fi_data %>%
  mutate(
    Currents_List = map(Currents_List, ~ fromJSON(gsub("nan", "null", gsub("'", '"', .), ignore.case=TRUE))),
    Firing_Rates_List = map(Firing_Rates_List, ~ fromJSON(gsub("nan", "null", gsub("'", '"', .), ignore.case=TRUE)))
  ) %>%
  unnest(cols = c(Currents_List, Firing_Rates_List)) %>%
  rename(Current = Currents_List, FiringRate = Firing_Rates_List) %>%
  filter(!is.na(FiringRate)) %>%
  mutate(
    Subject = as.factor(Cell_ID),
    Genotype = as.factor(Genotype),
    Current = as.factor(Current)
  )

cat("\nN subjects:", length(unique(fi_long$Subject)), "\n")
cat("N current levels:", length(unique(fi_long$Current)), "\n")

# 2-way RM ANOVA
model_fi <- aov(FiringRate ~ Genotype + Genotype * Current + Error(Subject/Current), data = fi_long)

cat("\n2-way Repeated Measures ANOVA Results:\n")
anova_summary_fi <- summary(model_fi)
print(anova_summary_fi)

# Extract statistics
# Robust extraction function for specific stratum
extract_from_stratum <- function(summary_obj, stratum_pattern, term_name) {
  # Find the stratum matching the pattern
  stratum_idx <- grep(stratum_pattern, names(summary_obj))
  
  if (length(stratum_idx) > 0) {
    stratum <- summary_obj[[stratum_idx[1]]][[1]]
    
    # Check if term exists in this stratum (using partial match but avoiding partial strings)
    # Using exact match on trimmed names
    row_idx <- which(trimws(rownames(stratum)) == term_name)
    
    if (length(row_idx) > 0) {
      f_val <- stratum[row_idx, "F value"]
      p_val <- stratum[row_idx, "Pr(>F)"]
      df1 <- stratum[row_idx, "Df"]
      
      # Use the residuals from the SAME stratum for df2
      res_idx <- which(trimws(rownames(stratum)) == "Residuals")
      if (length(res_idx) > 0) {
        df2 <- stratum[res_idx, "Df"]
      } else {
        df2 <- NA
      }
      return(list(F = f_val, P = p_val, df1 = df1, df2 = df2))
    }
  }
  return(list(F = NA, P = NA, df1 = NA, df2 = NA))
}

# F-I Curve Stats
# Genotype -> Error: Subject
b_stats <- extract_from_stratum(anova_summary_fi, "Error: Subject", "Genotype")
genotype_f <- b_stats$F
genotype_p <- b_stats$P
genotype_df1 <- b_stats$df1
genotype_df2 <- b_stats$df2

# Current & Interaction -> Error: Subject:Current (or similar within-subject error)
# Note: grep pattern "Subject:" catches "Subject:Current" or similar
w_stats_curr <- extract_from_stratum(anova_summary_fi, "Subject:", "Current")
current_f <- w_stats_curr$F
current_p <- w_stats_curr$P
current_df1 <- w_stats_curr$df1
current_df2 <- w_stats_curr$df2

w_stats_int <- extract_from_stratum(anova_summary_fi, "Subject:", "Genotype:Current")
interaction_f <- w_stats_int$F
interaction_p <- w_stats_int$P
interaction_df1 <- w_stats_int$df1
interaction_df2 <- w_stats_int$df2

# Overall mean firing rate per genotype (across all current steps)
mean_fi <- fi_long %>%
  group_by(Genotype) %>%
  summarise(
    Mean_FiringRate = mean(FiringRate, na.rm = TRUE),
    SEM_FiringRate  = sd(FiringRate, na.rm = TRUE) / sqrt(n()),
    N_cells         = n_distinct(Subject),
    .groups = 'drop'
  )

fi_anova_results <- data.frame(
  Term = c("Genotype", "Current", "Genotype:Current"),
  F_value = c(genotype_f, current_f, interaction_f),
  df1 = c(genotype_df1, current_df1, interaction_df1),
  df2 = c(genotype_df2, current_df2, interaction_df2),
  p_value = c(genotype_p, current_p, interaction_p),
  significance = c(sig_func(genotype_p), sig_func(current_p), sig_func(interaction_p))
)


cat("\nF-I Curve ANOVA Summary:\n")
print(fi_anova_results)


# ============================================================================
# PART 3: ISI ADAPTATION 2-WAY REPEATED MEASURES ANOVA
# ============================================================================
cat("\n================================================================================\n")
cat("PART 3: ISI ADAPTATION 2-WAY REPEATED MEASURES ANOVA\n")
cat("================================================================================\n")

# Reshape ISI data
isi_long <- fi_data %>%
  mutate(
    ISI_Times_List = map(ISI_Times_List, ~ {
      json_str <- gsub("nan", "null", gsub("'", '"', .), ignore.case = TRUE)
      # Handle empty strings
      if (is.na(json_str) || nchar(trimws(json_str)) == 0) return(NA)
      
      tryCatch({
          parsed <- fromJSON(json_str)
          if (is.null(parsed) || length(parsed) == 0) return(NA)
          if (is.list(parsed)) as.numeric(unlist(parsed)) else as.numeric(parsed)
      }, error = function(e) return(NA))
    })
  ) %>%
  unnest(cols = c(ISI_Times_List)) %>%
  group_by(Cell_ID) %>%
  mutate(Spike_Number = row_number() + 1) %>%  # ISI between spike 1-2 is for spike 2
  ungroup() %>%
  filter(Spike_Number >= 2 & Spike_Number <= 6) %>%  # Only spikes 2-6 like in the plot
  rename(ISI = ISI_Times_List) %>%
  filter(!is.na(ISI)) %>%
  mutate(
    Subject = as.factor(Cell_ID),
    Genotype = as.factor(Genotype),
    Spike_Number = as.factor(Spike_Number)
  )

cat("\nN subjects:", length(unique(isi_long$Subject)), "\n")
cat("Spike numbers:", paste(unique(isi_long$Spike_Number), collapse=", "), "\n")
cat("\nN observations per genotype:\n")
print(table(isi_long$Genotype))

# 2-way RM ANOVA for ISI
model_isi <- aov(ISI ~ Genotype * Spike_Number + Error(Subject/Spike_Number), data = isi_long)

cat("\n2-way Repeated Measures ANOVA Results:\n")
anova_summary_isi <- summary(model_isi)
print(anova_summary_isi)

# Extract statistics
# Extract statistics - Genotype from Subject stratum
b_stats_isi <- extract_from_stratum(anova_summary_isi, "Error: Subject", "Genotype")
geno_f_isi <- b_stats_isi$F
geno_p_isi <- b_stats_isi$P
geno_df1_isi <- b_stats_isi$df1
geno_df2_isi <- b_stats_isi$df2

# Spike & Interaction from Subject:Spike_Number stratum
w_stats_spike <- extract_from_stratum(anova_summary_isi, "Subject:", "Spike_Number")
spike_f_isi <- w_stats_spike$F
spike_p_isi <- w_stats_spike$P
spike_df1_isi <- w_stats_spike$df1
spike_df2_isi <- w_stats_spike$df2

w_stats_int_isi <- extract_from_stratum(anova_summary_isi, "Subject:", "Genotype:Spike_Number")
int_f_isi <- w_stats_int_isi$F
int_p_isi <- w_stats_int_isi$P
int_df1_isi <- w_stats_int_isi$df1
int_df2_isi <- w_stats_int_isi$df2

isi_anova_results <- data.frame(
  Term = c("Genotype", "Spike_Number", "Genotype:Spike_Number"),
  F_value = c(geno_f_isi, spike_f_isi, int_f_isi),
  df1 = c(geno_df1_isi, spike_df1_isi, int_df1_isi),
  df2 = c(geno_df2_isi, spike_df2_isi, int_df2_isi),
  p_value = c(geno_p_isi, spike_p_isi, int_p_isi),
  significance = c(sig_func(geno_p_isi), sig_func(spike_p_isi), sig_func(int_p_isi))
)

cat("\nISI Adaptation ANOVA Summary:\n")
print(isi_anova_results)


# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
cat("\n================================================================================\n")
cat("SAVING RESULTS\n")
cat("================================================================================\n")

# Combine all results
# Combine all results
# Empty placeholder for slope_result (slope analysis moved to Python/midpoint)
slope_result <- data.frame(
  Analysis = character(0), Comparison = character(0),
  Test = character(0), Statistic = numeric(0),
  p_value = numeric(0), significance = character(0)
)

# Mean firing rate summary rows
mean_fi_rows <- mean_fi %>%
  mutate(
    Analysis   = "F-I Curve",
    Test       = "Mean Firing Rate",
    Comparison = paste("Overall Mean:", Genotype),
    Statistic  = Mean_FiringRate,
    p_value    = NA_real_,
    significance = NA_character_
  ) %>%
  select(Analysis, Comparison, Test, Statistic, p_value, significance,
         Mean_FiringRate, SEM_FiringRate, N_cells)

cat("\nMean Firing Rate Summary:\n")
print(mean_fi %>% mutate(across(where(is.numeric), ~ round(., 2))))

all_results <- bind_rows(
  fi_anova_results %>% mutate(Analysis = "F-I Curve", Test = "2-way RM ANOVA", Comparison = Term, .before = 1) %>% select(-Term),
  isi_anova_results %>% mutate(Analysis = "ISI Adaptation", Test = "2-way RM ANOVA", Comparison = Term, .before = 1) %>% select(-Term),
  mean_fi_rows
)

# Determine output directory
output_dir <- "paper_data/Firing_Rate/"
if (!dir.exists("paper_data")) {
  output_dir <- "../paper_data/Firing_Rate/"
}

# Ensure directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(all_results, paste0(output_dir, "FI_ISI_Stats_Complete.csv"), row.names = FALSE)
cat(sprintf("\n✓ Saved complete results to: %sFI_ISI_Stats_Complete.csv\n", output_dir))

# Also save individual CSVs for backward compatibility
# write.csv(slope_result, paste0(output_dir, "FI_Slope_Stats.csv"), row.names = FALSE)
write.csv(fi_anova_results, paste0(output_dir, "FI_Curve_2way_ANOVA.csv"), row.names = FALSE)
write.csv(isi_anova_results, paste0(output_dir, "ISI_Adaptation_2way_ANOVA.csv"), row.names = FALSE)

cat("✓ Saved individual results:\n")
# cat(sprintf("  - %sFI_Slope_Stats.csv\n", output_dir))
cat(sprintf("  - %sFI_Curve_2way_ANOVA.csv\n", output_dir))
cat(sprintf("  - %sISI_Adaptation_2way_ANOVA.csv\n", output_dir))

cat("\n================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================================\n")
