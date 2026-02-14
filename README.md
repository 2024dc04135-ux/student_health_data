**Problem Statement:**

Predict student health risk levels (Low, Moderate, High) using physiological, lifestyle, and academic features.

**Dataset Description:**

•	Source: Student health dataset (custom/prepared).

•	Features: Age, Gender, Heart Rate, Blood Pressure, Stress Levels, Physical Activity, Sleep Quality, Mood, Study Hours, Project Hours, Family Members.

•	Target: Health_Risk_Level.

**Models Used:**

•	Logistic Regression

•	Decision Tree

•	KNN

•	Naive Bayes

•	Random Forest

•	XGBoost

 **Comparison Table with the evaluation metrics:**

|Model|	Accuracy|	AUC_SCORE| Precision|	Recall|	F1|	MCC|
|-------|-------|-------|-------|-------|-------|-------|
|Logistic Regression |0.795	|0.935668	|0.776353	|0.686104	|0.717568	|0.61106|
|Decision Tree	|0.98	|0.97239	|0.989333	|0.961658	|0.974602	|0.96398|
|KNN	|0.685	|0.794681	|0.704873	|0.49278	|0.519411	|0.364098|
|Naive Bayes	|0.735	|0.94684	|0.703292	|0.557053	|0.583288	|0.48364|
|Random Forest	|0.945	|0.996753	|0.967463	|0.888741	|0.917952	|0.901633|
|XGBoost	|1	|1	|1	|1	|1	|1|

 **observations on the performance of each model:**

|Model|	Observation|
|--|--|
|Logistic Regression	|Good baseline, moderate accuracy.|
|Decision Tree	|Overfits slightly, lower generalization.|
|KNN	|Sensitive to scaling, moderate performance.|
|Naive Bayes	|Fast but weaker on mixed features.|
|Random Forest	|Strong ensemble, robust accuracy.|




