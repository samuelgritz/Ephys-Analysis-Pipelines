# Comprehensive F-I and ISI Statistics
# Includes: F-I curve 2-way ANOVA, F-I slope comparison, ISI adaptation 2-way ANOVA
library(tidyverse)
library(lme4)
library(lmerTest)

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
    # Derive Animal from Cell_ID: strip _cN suffix (each date = one mouse)
    Animal = as.factor(sub("_c[0-9]+$", "", as.character(Cell_ID))),
    Genotype = as.factor(Genotype),
    Current = as.factor(Current)
  )

cat("\nN subjects (cells):", length(unique(fi_long$Subject)), "\n")
cat("N animals (dates):", length(unique(fi_long$Animal)), "\n")
cat("N current levels:", length(unique(fi_long$Current)), "\n")

# 2-way LME with nested random effects (Animal/Subject)
model_fi <- lmer(FiringRate ~ Genotype * Current + (1 | Animal/Subject), data = fi_long)

cat("\nLME ANOVA Results (Type III, Satterthwaite):\n")
anova_result_fi <- anova(model_fi)
print(anova_result_fi)

# Extract statistics from lmerTest ANOVA table
anova_df_fi <- as.data.frame(anova_result_fi)
anova_df_fi$Term <- rownames(anova_df_fi)

# Map term names for output
term_map <- c("Genotype" = "Genotype", "Current" = "Current", "Genotype:Current" = "Genotype:Current")

fi_anova_results <- data.frame(
  Term = character(0),
  F_value = numeric(0),
  df1 = numeric(0),
  df2 = numeric(0),
  p_value = numeric(0),
  significance = character(0)
)

for (term_name in c("Genotype", "Current", "Genotype:Current")) {
  if (term_name %in% rownames(anova_df_fi)) {
    row_data <- anova_df_fi[term_name, ]
    fi_anova_results <- rbind(fi_anova_results, data.frame(
      Term = term_name,
      F_value = row_data[["F value"]],
      df1 = row_data[["NumDF"]],
      df2 = row_data[["DenDF"]],
      p_value = row_data[["Pr(>F)"]],
      significance = sig_func(row_data[["Pr(>F)"]])
    ))
  }
}


# Overall mean firing rate per genotype (across all current steps)
mean_fi <- fi_long %>%
  group_by(Genotype) %>%
  summarise(
    Mean_FiringRate = mean(FiringRate, na.rm = TRUE),
    SEM_FiringRate  = sd(FiringRate, na.rm = TRUE) / sqrt(n()),
    N_cells         = n_distinct(Subject),
    .groups = 'drop'
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
    # Derive Animal from Cell_ID: strip _cN suffix (each date = one mouse)
    Animal = as.factor(sub("_c[0-9]+$", "", as.character(Cell_ID))),
    Genotype = as.factor(Genotype),
    Spike_Number = as.factor(Spike_Number)
  )

cat("\nN subjects (cells):", length(unique(isi_long$Subject)), "\n")
cat("N animals (dates):", length(unique(isi_long$Animal)), "\n")
cat("Spike numbers:", paste(unique(isi_long$Spike_Number), collapse=", "), "\n")
cat("\nN observations per genotype:\n")
print(table(isi_long$Genotype))

# 2-way LME with nested random effects (Animal/Subject)
model_isi <- lmer(ISI ~ Genotype * Spike_Number + (1 | Animal/Subject), data = isi_long)

cat("\nLME ANOVA Results (Type III, Satterthwaite):\n")
anova_result_isi <- anova(model_isi)
print(anova_result_isi)

# Extract statistics from lmerTest ANOVA table
anova_df_isi <- as.data.frame(anova_result_isi)

isi_anova_results <- data.frame(
  Term = character(0),
  F_value = numeric(0),
  df1 = numeric(0),
  df2 = numeric(0),
  p_value = numeric(0),
  significance = character(0)
)

for (term_name in c("Genotype", "Spike_Number", "Genotype:Spike_Number")) {
  if (term_name %in% rownames(anova_df_isi)) {
    row_data <- anova_df_isi[term_name, ]
    isi_anova_results <- rbind(isi_anova_results, data.frame(
      Term = term_name,
      F_value = row_data[["F value"]],
      df1 = row_data[["NumDF"]],
      df2 = row_data[["DenDF"]],
      p_value = row_data[["Pr(>F)"]],
      significance = sig_func(row_data[["Pr(>F)"]]),
      stringsAsFactors = FALSE
    ))
  }
}

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
