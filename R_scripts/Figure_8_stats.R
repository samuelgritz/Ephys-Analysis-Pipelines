library(dplyr)

# ==============================================================================
# DATA LOADING
# ==============================================================================

# Determine data root to resolve path issues
if (file.exists("paper_data")) {
  data_root <- "paper_data"
} else if (file.exists("../paper_data")) {
  data_root <- "../paper_data"
} else {
  stop("Could not find 'paper_data' directory.")
}

df_path <- file.path(data_root, "Plateau_data/Plateau_Delta_GIRK.csv")

cat(paste("Attempting to load data from:", df_path, "\n"))

if (!file.exists(df_path)) {
  stop(paste("Delta CSV not found at:", df_path))
}

# Try reading
tryCatch({
  df <- read.csv(df_path)
  cat(paste("Successfully loaded data. Rows:", nrow(df), "\n"))
}, error = function(e) {
  stop(paste("Failed to read CSV:", e$message))
})

# ==============================================================================
# ANALYSIS
# ==============================================================================

# Initialize results DF
results <- data.frame()

# Function to run stats per drug/pathway
run_stats <- function(df_sub, drug, path) {
  # Filter
  sub <- df_sub %>% filter(Drug == drug, Pathway == path)
  
  # Check sample size
  n_wt <- sum(sub$Genotype == "WT")
  n_gnb1 <- sum(sub$Genotype == "GNB1")
  
  if (n_wt >= 3 & n_gnb1 >= 3) {
    # Mann-Whitney U Test (Wilcoxon Rank Sum) comparing Genotypes on Delta
    # Enforced as per statistical standardization plan
    test <- wilcox.test(Delta_Area ~ Genotype, data = sub)
    
    return(data.frame(
      Drug = drug,
      Pathway = path,
      Comparison = "WT vs GNB1 (Delta)",
      p_value = test$p.value,
      test_stat = test$statistic,
      test_type = "Mann-Whitney U",
      WT_mean = mean(sub$Delta_Area[sub$Genotype=="WT"]),
      GNB1_mean = mean(sub$Delta_Area[sub$Genotype=="GNB1"]),
      WT_n = n_wt,
      GNB1_n = n_gnb1
    ))
  } else {
    return(NULL)
  }
}

# Run loop
for (drug in c("ML297", "ETX")) {
  for (path in unique(df$Pathway)) {
    res <- run_stats(df, drug, path)
    if (!is.null(res)) {
      results <- rbind(results, res)
    }
  }
}

# ==============================================================================
# SAVING
# ==============================================================================

# Save results
out_path <- file.path(data_root, "Plateau_data/Stats_Results_Figure_8.csv")
write.csv(results, out_path, row.names = FALSE)
cat(paste("✓ Statistics saved to:", out_path, "\n"))
print(results)
