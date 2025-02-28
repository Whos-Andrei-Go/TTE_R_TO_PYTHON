# Load the TrialEmulation package
library("TrialEmulation")
library(this.path)

# Set working directory to the script's directory
setwd(this.path::this.dir())

# Load the data_censored dataset
data("data_censored")

# Display the first few rows of the data
head(data_censored)

# Define the file path (saved in the script's directory inside "csv-files")
current_dir <- getwd()

# Define the file name
file_name <- "data_censored.csv"

# Create the full file path
file_path <- file.path(current_dir, file_name)

# Write the data to a CSV file
write.csv(data_censored, file = file_path, row.names = FALSE)

# Confirm the file is saved
cat("CSV file saved at:", file_path)
