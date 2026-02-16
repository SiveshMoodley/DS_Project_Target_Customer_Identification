# Identifying Potential Target Customers for Educational Product Exxtra Learn

## Project Overview
- Built a classification model to predict which ExtraaLearn leads are most likely to convert into paid customers, enabling better sales/marketing prioritisation.
- Engineered predictive features using structured preprocessing pipelines, including one-hot encoding of categorical variables, numerical feature scaling, and class imbalance handling (SMOTE) to improve minority-class detection and model stability.
- Trained and optimised multiple classification algorithms (Decision Tree, Random Forest, Gradient Boosting, and XGBoost), leveraging cross-validation and performance comparison to select the best-performing lead conversion model.

## Code and Resources
Python Version: 3.10

Packages: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost 

Python Requirements: pip install -r requirements.txt

## Dataset
Source: MIT Professional Education 

Type: Educational / Synthetic Dataset

The dataset represents potential customer lead data collected by Extraa Learn, an educational platform, with the objective of identifying which prospective users are most likely to convert into paid customers. The dataset simulates a real-world marketing analytics scenario where organisations seek to optimise customer acquisition and sales targeting.

The dataset contains lead-level information including:
- ID of the lead (ID)
- Age of the lead (age)
- Occupation of the lead (current_occupation)
- How the lead first interacted with Extraa Learn - website/mobile app (first_interaction)
- What percentage of profile has been filled by the lead on the website/mobile app, either Low - 0-50%, Medium - 50-75%, High 75-100% (profile_completed)
- How many times has a lead visited the website (website_visits)
- Total time spent on the website (time_spent_on_website)
- Average number of pages on the website viewed during the visits (page_views_per_visit)
- Last interaction between the lead and Extraa Learn representative - email/phone/website (last_activity)
- Flag indicating whether the lead had seen the ad of ExtraaLearn in the Newspaper (print_media_type1)
- Flag indicating whether the lead had seen the ad of ExtraaLearn in the Magazine (print_media_type2)
- Flag indicating whether the lead had seen the ad of ExtraaLearn on the digital platforms (digital_media)
- Flag indicating whether the lead had heard about ExtraaLearn in the education channels like online forums, discussion threads, educational websites, etc (educational_channels)
- Flag indicating whether the lead had heard about ExtraaLearn through reference (referral)
- Flag indicating whether the lead was converted to a paid customer or not (status)

The data is intended for educational and model development purposes. No real personal or sensitive customer information is included.

## Data Preprocessing
The data was prepared to ensure it was ready for the model. I conducted the following process:

- Checked dataset shape, feature data types, duplicate rows, and missing values to validate overall data quality
- Dropped the unique identifier column (ID) as it carries no predictive value and separated the target (status) from the feature set
- Split the data into training and test sets, with a 20% test size
- Applied StandardScaler to numerical features to normalise magnitude and improve model convergence
- Applied OneHotEncoder to categorical features to convert categories into machine-readable format

The data was prepared and split into training and test sets, with a 20% test size.

## EDA
Conducted exploratory data analysis to understand feature distributions, class balance, and behavioural patterns associated with conversion. Key observations include:
- The target variable (status) is moderately imbalanced, with non-converted leads forming the majority class, necessitating recall-focused evaluation
- Time spent on website is the strongest predictor of conversion, showing the highest positive association with the target variable
- Engagement-based indicators (profile completion and website activity metrics) demonstrate meaningful distributional differences between converted and non-converted leads, indicating strong discriminatory power
- Lead origin indicators (digital media, referral, print media, and educational channels) and first interaction channel show statistically meaningful differences in conversion rates, with referral and website-origin leads exhibiting higher likelihood of conversion

## Model Building
Multiple classification algorithms were trained and evaluated to identify the most effective lead conversion model:
- Decision Tree – baseline non-linear model to capture interaction effects and hierarchical decision rules
- Random Forest – ensemble bagging approach to reduce overfitting and improve generalisation
- Gradient Boosting – sequential boosting model designed to minimise residual errors and improve predictive accuracy
- XGBoost – optimised gradient boosting framework built for high-performance structured data modeling

To address class imbalance and improve recall of converted leads, SMOTE was applied to the training data for selected models.

Hyperparameter tuning was conducted using cross-validation to optimise model performance prior to final evaluation on the test set.

## Model Performance
Models were evaluated using Accuracy, ROC-AUC, Precision, and Recall, with particular emphasis on recall for the converted-lead class (status = 1), as correctly identifying high-potential customers is the primary business objective.

- Decision Tree achieved Accuracy: 0.86, ROC-AUC: 0.93, with Recall (1): 0.73 and Precision (1): 0.79
- Random Forest delivered strong overall performance (Accuracy: 0.87, ROC-AUC: 0.93) with Recall (1): 0.75 and Precision (1): 0.81
- Gradient Boosting matched Random Forest closely (Accuracy: 0.87, ROC-AUC: 0.93) with Recall (1): 0.75 and Precision (1): 0.80
- XGBoost produced the best base-model minority capture with Accuracy: 0.87, ROC-AUC: 0.93, Recall (1): 0.76, Precision (1): 0.80
- SMOTE Balanced Random Forest improved converted-lead detection, achieving Accuracy: 0.86, ROC-AUC: 0.93, with Recall (1): 0.82 and Precision (1): 0.75 (best recall uplift with acceptable trade-off)

## Project Evaluation
- Identified key predictors of conversion, notably behavioural engagement metrics such as time spent on website and profile completion, confirming that intent-based signals are stronger drivers of conversion than purely demographic attributes
- Addressed dataset class imbalance using SMOTE, improving converted-lead recall while accepting lower precision — an appropriate trade-off in customer acquisition where failing to identify high-potential leads carries greater business cost
- Determined XGBoost and SMOTE-balanced Random Forest as the optimal models, achieving strong class discriminatory performance and improved minority-class detection for operational deployment
- Highlighted limitations including class imbalance, potential overfitting risk in boosted models, and possible behavioural drift over time, underscoring the need for monitoring and periodic retraining
- Recommended future enhancements including cost-sensitive learning, threshold optimisation aligned to marketing profitability objectives, and incorporation of additional behavioural features, like engagement frequency
