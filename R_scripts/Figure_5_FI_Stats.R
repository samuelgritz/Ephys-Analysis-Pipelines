# Figure 5 Baclofen FI Difference Statistics
# Analysis: Repeated Measures ANOVA (Genotype x Current) on Firing Rate Reduction

# Load Data using Base R
data_path <- "../paper_data/gabab_analysis/Baclofen_FI_Difference.csv"
if (!file.exists(data_path)) {
  data_path <- "paper_data/gabab_analysis/Baclofen_FI_Difference.csv" # Fallback
}

if (!file.exists(data_path)) {
  stop(paste("Error: Data file not found at", data_path))
}

df <- read.csv(data_path)

cat("==============================================================================\n")
cat("Figure 5: Baclofen-Induced Reduction Stats (R Script)\n")
cat("==============================================================================\n")
cat(paste("Total Data Points:", nrow(df), "\n"))
cat(paste("Unqiue Cells:", length(unique(df$Cell_ID)), "\n"))

# Ensure factors
df$Genotype <- factor(df$Genotype, levels = c("WT", "GNB1"))
df$Current_pA_Factor <- factor(df$Current_pA)
df$Subject <- factor(df$Cell_ID)

# Try using lmerTest for Mixed Model (Preferred)
use_lmer <- FALSE
if (require("lmerTest", quietly = TRUE)) {
  use_lmer <- TRUE
  cat("\n[Method] Using Linear Mixed-Effects Model (lmerTest)...\n")
  
  model <- lmer(Difference_FR ~ Genotype + Genotype * Current_pA_Factor + (1 | Subject), data = df)
  
  cat("\n--- ANOVA Table (Type III) ---\n")
  ano <- anova(model, type = 3)
  print(ano)
  
  # Save
  write.csv(as.data.frame(ano), "../paper_data/gabab_analysis/Figure_5_FI_Stats_R_LMM.csv")
  
} else {
  cat("\n[Method] 'lmerTest' not found. Using Standard Repeated Measures ANOVA (aov)...\n")
  cat("Model: Difference ~ Genotype * Current + Error(Subject/Current)\n")
  
  # Repeated Measures ANOVA
  # Error term (Subject/Current_pA_Factor) accounts for repeated measures
  # model <- aov(Difference_FR ~ Genotype * Current_pA_Factor + Error(Subject/Current_pA_Factor), data = df)
  
  cat("\n--- ANOVA Summary ---\n")
  #do an anova on the model with no error term
  model2 <- aov(Difference_FR ~ Genotype * Current_pA_Factor, data = df)
  print(anova(model2))
  


  
  # Extract p-values for CSV saving (Basic extraction)
  output_dir <- dirname(data_path)
  outfile <- file.path(output_dir, "Figure_5_FI_Stats_R_ANOVA.txt")
  
  capture.output(summary(model), file = outfile)
  cat(paste("\nStats summary saved to:", outfile, "\n"))
}

cat("\n==============================================================================\n")

