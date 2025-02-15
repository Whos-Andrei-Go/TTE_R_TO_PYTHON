# Load the TrialEmulation package
library("TrialEmulation")

# Load the data_censored dataset
data("data_censored")

# Display the first few rows of the data
head(data_censored)

# Specify the folder and file name
base_folder <- file.path(path.expand("~"), "USC", "2nd Year", "3203", "TTE_R_TO_PYTHON")
file_name <- "data_censored.csv"
file_path <- file.path(base_folder, file_name)

# Write the data to a CSV file
write.csv(data_censored, file = file_path, row.names = FALSE)

# Confirm the file is saved
cat("CSV file saved at:", file_path)
