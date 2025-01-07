# Data Preparation: Since our dataset did not have any duplicate, incorrect, or missing values, we did not essentially have to fix the data quality issues. Instead, we just transformed some of our categorical attributes like Scholarship_holder and Gender to factor attributes in order to bring our dataset to the appropriate processing format in order to train our logistic regression model.

# Data Preprocessing: For this phase of our data mining pipeline, we conducted feature selection by discarding certain attributes like GDP, debtor, semester-wise curricular units from our dataset that were not relevant for our project's objective or our machine learning model.


# Set the working directory to the local path of the target file (as discussed in class)
setwd('/Users/shaya/Library/CloudStorage/OneDrive-HigherEducationCommission/ASU - Data Science, Analytics, and Engineering (M.Sc.)/Spring 2024/HSE 531 Data Analytics for Modeling Human Subject Data/Project')


# Read in the dropout data file into a data frame named "dropout.df"
dropout.df = read.csv(file = "dropout.csv")

# Print the data frame summary to check the structure and contents of the data
summary(dropout.df)


library(ggplot2)

##############################################################################################################
# Exploratory Data Analysis (EDA)

# Create histogram of Age_at_enrollment showing the distribution of the ages of the students
ggplot(dropout.df, aes(x = Age_at_enrollment)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age at Enrollment",
       x = "Age at Enrollment",
       y = "Frequency") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# Histogram of the Distribution of Inflation Rate
# Not sure about this one (??)
ggplot(dropout.df, aes(x = Inflation_rate)) +
  geom_histogram(binwidth = 0.1, fill = "pink", color = "black", alpha = 1) +
  labs(title = "Distribution of Inflation Rate", x = "Inflation Rate", y = "Frequency")


# Create stacked bar plot of Course Enrollment by Gender
ggplot(dropout.df, aes(x = Course_Name, fill = Gender)) +
  geom_bar(position = "stack") +
  labs(title = "Course Enrollment by Gender",
       x = "Course",
       y = "Count",
       fill = "Gender") +
  scale_fill_manual(values = c("Male" = "navy", "Female" = "magenta")) +
  theme_minimal()


# Create pie chart of Nationality Distribution (Incorrect graph)
ggplot(dropout.df, aes(x = "", fill = Nationality)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Nationality Distribution") +
  scale_fill_manual(values = rainbow(length(unique(dropout.df$Nationality)))) +
  theme_minimal() +
  theme(legend.position = "right")



# Bar plot showing distribution of Previous Qualification of students. The bar graph shows class 1 as the Previous Qualification with the highest frequency. 1, in our dataset, denotes secondary education. The second highest class is 39 which is technological specialization courses.
ggplot(dropout.df, aes(x = factor(Previous_qualification))) +
  geom_bar(fill = "violet", color = "black") +
  labs(title = "Previous Qualification Distribution",
       x = "Previous Qualification",
       y = "Count") +
  theme_minimal()


# Stacked bar plot of Target variable by Previous Qualification distribution. This graph shows that for students with secondary education (class 1) as their previous qualification, ones with 'Graduate' class label have the highest frequency of around 2000, followed by 'Dropouts' of around 1000 counts, and lastly by 'Enrolled' students with a count of around 700.
ggplot(dropout.df, aes(x = factor(Previous_qualification), fill = Target)) +
  geom_bar() +
  labs(title = "Distribution of Target Variable by Previous Qualification",
       x = "Previous Qualification",
       y = "Count",
       fill = "Target") +
  theme_minimal()

# Plot 1 (included in our preliminary data analysis discussion post)

library(dplyr)

# Calculate dropout rates by Gender and Course_Name
dropout_rates <- dropout.df %>%
  group_by(Course_Name, Gender) %>%
  summarise(dropout_rate = mean(Target == "Dropout") * 100)  # Calculate dropout rate as a percentage

# Heatmap for the distribution of dropout rates by Gender and Course
ggplot(dropout_rates, aes(x = Gender, y = Course_Name, fill = dropout_rate)) +
  geom_tile() +
  scale_fill_gradient(low = "skyblue", high = "red", name = "Dropout Rate (%)") +
  labs(title = "Distribution of Dropout Rates by Gender and Course",
       x = "Gender",
       y = "Course Name") +
  theme_minimal()

# The graph illustrates variations in dropout rates across different courses by gender. Notably, women exhibit a higher dropout rate across all courses. Additionally, men demonstrate the highest dropout percentage in Informatics Engineering. Similarly, women show the highest dropout rate in Biofuel Production Technologies. Moreover, both men and women experience the lowest dropout rates in Nursing. This graph highlights the importance of considering gender as a predictor of dropout behavior and shows courses where interventions and support are needed. 

# Plot 2 (included in our preliminary data analysis discussion post)

library(tidyverse)

# Convert Scholarship_holder categorical attribute to factor with labels (Data Transformation)
dropout.df$Scholarship_holder <- factor(dropout.df$Scholarship_holder, labels = c("No", "Yes"))
#df$Dropped_Out <- as.factor(df$Dropped_Out)
# Calculate the dropout rate
dropout_rates <- dropout.df %>%
  group_by(Scholarship_holder) %>%
  summarise(Dropout_Rate = mean(Target == "Dropout")) %>%
  ungroup()

# Multiply Dropout_Rate by 100 to convert it to percentage
dropout_rates$Dropout_Percentage <- dropout_rates$Dropout_Rate * 100

# Now, plotting the bar chart
# A bar graph exhibiting comparison of dropout rates by scholarship status
ggplot(dropout_rates, aes(x = Scholarship_holder, y = Dropout_Percentage, fill = Scholarship_holder)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("No" = "red", "Yes" = "green")) +
  labs(title = "Comparison of Dropout Rates by Scholarship Holder Status",
       x = "Scholarship Holder",
       y = "Dropout Percentage",
       fill = "Scholarship Holder") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# In this analysis, we aimed to examine socioeconomic variables, focusing specifically on the attribute "Scholarship_holder" in relation to dropout rates. The dropout percentage was calculated based on the proportion of the label "Dropout" from the "Target" attribute. Our findings reveal that students who hold scholarships are less likely to dropout, with a dropout percentage of around 12.5%, compared to non-scholarship holders, who exhibit a dropout percentage of around 38.5%. Such a high disparity between the dropout rates of the two groups strongly implies the influence of financial characteristics on students’ dropout risk. By examining how scholarship holders correlate with dropout rates, educators, policymakers, and stakeholders can better tailor interventions for early warning systems to address at-risk students.

##############################################################################################################
# Model Building (Logistic Regression)

# Recode 'Target' variable so that it holds either 1s (Dropout) or 0s (Graduate or Enrolled) (Data Transformation)
dropout.df$Target <- ifelse(dropout.df$Target == 'Dropout', 1, 0)

# Convert categorical variables to factors (Data Transformation)
dropout.df$Gender <- as.factor(dropout.df$Gender)
levels(dropout.df$Gender) <- c("1", "2")

dropout.df$Scholarship_holder <- as.factor(dropout.df$Scholarship_holder)
levels(dropout.df$Scholarship_holder) <- c("1", "0")

# Split the dataset into training and testing sets by the ratio of 7:3
set.seed(123)  # For reproducibility of the random sampling process. Sets the seed for the random number generator
train_index <- sample(1:nrow(dropout.df), 0.7 * nrow(dropout.df))
train_data <- dropout.df[train_index, ]
test_data <- dropout.df[-train_index, ]

# Select relevant features and interactions to build the logistic regression model on our training data
lr.model <- glm(Target ~ Mother_qualification * Father_qualification + Marital_status * Inflation_rate + Gender * Nationality * Age_at_enrollment + Previous_qualification + Tuition_fees_up_to_date + Scholarship_holder + Age_at_enrollment * Tuition_fees_up_to_date,
                family = binomial(link = "logit"),
                data = train_data)

# Install the packages
install.packages("caret")
install.packages("lme4")

# Load the relevant libraries
library(caret)
library(sjPlot)
library(lme4)

# Generate predictions using the logistic regression model (lr.model) on the test dataset (test_data)
predictions <- predict(lr.model, test_data, type = "response")
# Convert predicted probabilities to binary classes based on a threshold of 0.5
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
print(predicted_classes)


# Model Evaluation

# Calculate accuracy by comparing predicted classes with actual target values in the test dataset
accuracy <- mean(predicted_classes == test_data$Target)
# Our model gives an accuracy of 76.28% but to further realize the performance of our model, our evaluation utilized various statistical metrics such as p-values, precision, recall, and the ROC AUC value to ensure robustness in our conclusions.

summary(lr.model)

# Generate a table model summary to view the most significant predictors and their respective p-values
tab_model(lr.model)
# The bolded p-values we get here, which are below the 5% threshold in the table, highlight the most significant predictors affecting the dropout risk.

# Compute the confusion matrix based on predicted classes and actual target values
conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(test_data$Target))

# Convert the confusion matrix to a data frame for plotting
conf_mat_df <- as.data.frame(as.table(conf_matrix$table))

# Rename the levels for clarity in the plot
levels(conf_mat_df$Prediction) <- c("Predicted Negative", "Predicted Positive")
levels(conf_mat_df$Reference) <- c("Actual Negative", "Actual Positive")

# Calculate precision and recall from the confusion matrix for further evaluations of our trained model
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
# Coming to Precision which is the accuracy over positive predicted class. The precision comes out to be around 75.13% which, in our context means that around 75% of the students were accurately predicted as dropouts out of the total number of students predicted as dropouts by our regression model. On the other hand, Recall, which is the accuracy over positive class or the true positive rate, is computed to be around 96.87% for our model, which implies that out of all the actual Dropout students, almost 97% of the students were accurately predicted as Dropouts by our model.

# Print precision and recall
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")

print(conf_matrix)

# Plot the confusion matrix heatmap
heatmap_plot <- ggplot(conf_mat_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") + # Use white lines to separate the tiles
  geom_text(aes(label = Freq), vjust = 1.5, color = "black", size = 5) + # Add text labels
  scale_fill_gradient(low = "white", high = "steelblue") + # Color gradient from white to steelblue
  labs(title = "Confusion Matrix Heatmap", x = "Actual Label", y = "Predicted Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) # Center the title

# Print the plot
print(heatmap_plot)

# Plot the ROC curve to visualize the model's performance across various thresholds.
library(pROC)
roc_curve <- roc(test_data$Target, predictions)
plot(roc_curve, main="ROC Curve", xlab="False Positive Rate (1 - Specificity)", ylab="True Positive Rate (Sensitivity)", col="#1c61b6")

# The area under the ROC curve (AUC) is a single scalar value that summarizes the overall performance of the model. The closer the AUC is to 1, the better the model is at distinguishing between students who will drop out and those who will not.
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
# Since we are getting an AUC of 0.802 which is closer to 1, we can infer that our model has a good measure of separability between positive and negative classes. It means our model can distinguish between dropout and non-dropout students quite effectively.