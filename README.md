# Predicting Student Dropout Risk for Educational Interventions
A data-driven project analyzing factors influencing student dropout rates and predicting student risk for early interventions.

This repository contains the **Predicting Student Dropout Risk** project, which uses data-driven approaches to identify factors influencing student dropout rates. By analyzing demographic, academic, and socioeconomic variables, the project aims to support targeted educational interventions that improve student retention.

## 📜 **Project Overview**

The project investigates how various factors—demographic, academic, and socioeconomic—impact dropout risk among undergraduate students. Through rigorous data preprocessing, exploratory data analysis (EDA), and logistic regression modeling, this study provides insights into key predictors of dropout risk, enabling institutions to design effective support strategies.

### **Objectives**
1. Identify specific combinations of demographic, academic, and socioeconomic factors most strongly associated with increased dropout risk.
2. Develop a predictive model using logistic regression to identify students at risk of dropping out.

---

## 🚀 **Key Highlights**

### **Technical Approach**
1. **Data Preprocessing**:
   - Cleaned dataset to ensure data integrity and reliability.
   - Transformed categorical variables into factor variables for modeling.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized and analyzed trends in dropout rates across demographic, academic, and socioeconomic attributes.
   - Identified disparities in dropout risk based on factors such as tuition fees, inflation rates, and prior qualifications.

3. **Predictive Modeling**:
   - Applied **logistic regression** to model dropout risk.
   - Evaluated model performance using metrics like **accuracy (76.28%)**, **precision (75.13%)**, **recall (96.87%)**, and **ROC AUC (0.8)**.

---

## 📊 **Key Findings**

1. **Significant Predictors of Dropout Risk**:
   - Age at enrollment.
   - Previous qualification.
   - Tuition fees up to date.
   - Scholarship holder status.
   - Inflation rate.
   - Combination of age and tuition fees.

2. **Insights**:
   - Students from diverse academic and socioeconomic backgrounds experience unique challenges affecting their dropout risk.
   - The interplay between demographic (e.g., age) and socioeconomic factors (e.g., tuition fees) significantly influences dropout risk.

---

## 🌟 **Impact and Implications**

The findings provide institutions with actionable insights to:
- Identify high-risk students early and allocate resources efficiently.
- Develop targeted interventions to improve student retention.
- Enhance long-term academic success through better support systems.

---

## 🛠️ **Technologies Used**

- **Programming Languages**: R
- **Libraries**: `caret`, `ggplot2`, `dplyr`
- **Methods**:
  - Data preprocessing and cleaning.
  - Logistic regression modeling.
  - Evaluation metrics (Accuracy, Precision, Recall, ROC AUC).

---

## 🔬 **Limitations**

1. **Variables**:
   - Absence of key variables such as accessibility, GPA, and ethnicity/race might limit model performance.
2. **Generality**:
   - Results and insights may not be universally applicable across all institutions.

---

## 💡 **Future Directions**

- Include additional variables like accessibility, displacement, and ethnicity/race to improve the model.
- Develop interventions tailored to address demographic-specific dropout risks.
- Explore advanced modeling techniques like ensemble learning for better predictions.

