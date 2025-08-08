### Introduction

This work explores the development of a diagnostic classification model for the purposes of early diagnosis of pancreatic cancer. In order to not overwhelm the length of the Jupyter notebook, Clinical Evidence, Explanations of Model Development and Explainable Analysis can be found in this readme. This file can be read in tandem with the associated Jupyter notebook. 

### Additional Code: 
pip install shap

pip install lime

pip install xgboost

### Clinical Evidence

Pancreatic cancer stands at the 7th highest cause of cancer related deaths.(Rawla et al., 2019) While it is less prevalent than other cancers, its high mortality rate, which is as low as 2% in some countries, is alarming.(McGuigan et al., 2018) This is largely due to late diagnosis of the cancer. (Rawla et al., 2019) In 85% of cases, the cancerous tumor are no longer surgically removable at the point of detection.(eBioMedicine, 2022) Majority of pancreatic cancer cases are pancreatic ductal adenocarcinomas (PDAC), which will henceforth be used interchangeably with pancreatic cancer in this work, for simplicity. (McGuigan et al., 2018) At early stages, the disease is largely asymptomatic, or displays less concerning symptoms such as dorsal pain or nausea, contributing to its low rate of early detection.(Partyka et al., 2023; Rawla et al., 2019) Early detection of pancreatic cancer is tricky,  many viable techniques are invasive and expensive to conduct for screening purposes.(Partyka et al., 2023) Thus, developing a viable, easy to conduct and inexpensive mode of detection is crucial.

Researchers have investigated various biofluid biomarkers to determine relationships between biomarker expressions and presence of pancreatic cancer.A blood biomarker, CA19-9, is the only known biomarker for pancreatic cancer, but it is not sensitive or specific enough to detect pancreatic cancer at earlier stages. .(eBioMedicine, 2022; Tatjana Crnogorac-Jurcevic, n.d.) However, findings from proteomic studies using mass spectrometry have distinguished three key biomarkers: LYVE1, REG1B and TFF1 as highly probable discriminators between healthy patients, patients with benign conditions and patients with pancreatic cancer.(Tatjana Crnogorac-Jurcevic, n.d.)(Radon et al., 2015) Given the symptomatic similarity between non-cancerous ailments and pancreatic cancer, distinguishing between benign conditions with similar symptoms and pancreatic cancer is also important. As such a key area for the development of an early detection tool are urinary biomarkers. (eBioMedicine, 2022) 

The dataset chosen for this work is obtained from a study by Debernandi and colleagues, “A combination of urinary biomarker panel and PancRISK score for earlier detection of pancreatic cancer: A case–control study”.(Debernardi et al., 2020) It contains 590 retrospectively obtained urine samples from 3 groups: Healthy Controls, Hepatobiliary diseases and Patients with Pancreatic Cancer. The samples were obtained in 2 cohorts, from multiple centers. The dataset was varied, to include other common malignancies in order to represent the variations in urinary biomarkers.(Debernardi et al., 2020) Below is a table explaining the data available in the dataset.

| **Column** | **Description** |
|------------|------------------|
| **Sample_ID** | Identifier for each sample. |
| **Patient Cohort** | The cohort the patient sample was obtained (two cohorts total). This data has no link to diagnosis and will be dropped in further analysis. |
| **Sample Origin** | The center where the sample was collected (four centers, with unequal sample sizes). This will also be dropped. |
| **Age** | Age of the patient; follows a normal distribution between 20–90 years. |
| **Sex** | Gender of the patient; fairly balanced distribution, with slightly more female samples than male. |
| **Diagnosis** | Diagnosis category, divided into three groups: (0) Healthy Controls, (1) Patients with hepatobiliary diseases, (2) Patients with Pancreatic Cancer. Originally 1,2,3, Recoded as 0, 1, 2 for XGBoost classifiers. |
| **Stage** | Cancer stage. Removed from analysis to prevent data leakage, since all pancreatic cancer samples would have a stage. |
| **Benign Sample Diagnosis** | Specific conditions diagnosed in patients with hepatobiliary diseases. Removed as it was not directly relevant to pancreatic cancer. |
| **Plasma CA19_9 (U/ml)** | A blood biomarker indicative of pancreatic cancer. Removed as it was only available for about half the dataset and pertains to plasma, not urine. |
| **Creatinine (mg/ml)** | A byproduct of amino acid metabolism, commonly used to assess kidney function (Asif et al., n.d.). Urine biomarker values are normalized against creatinine due to urine concentration variability (Tang et al., 2015). |
| **LYVE1 (ng/ml)** | Lymphatic Vessel Endothelial HA Receptor 1, a surface receptor expressed on lymphatic endothelial cells. Studies show decreased expression in pancreatic cancer tissues (Samir et al., 2024). |
| **REG1B (ng/ml)** | Regenerating Islet-derived 1 Beta, a protein similar to REG1A, with closely associated expression patterns (Radon et al., 2015; van Beelen Granlund et al., 2013). Used here in place of REG1A due to similar behavior. |
| **TFF1 (ng/ml)** | Trefoil Factor Family 1, a secretory protein maintaining gastrointestinal mucus membrane integrity. Differential expression is strongly linked to various cancers (Samir et al., 2024). |
| **REG1A (ng/ml)** | Regenerating Islet-derived 1 Alpha, a pancreatic protein regulating insulin secretion and pancreatic function (Samir et al., 2024). Increased expression is associated with pancreatic cancer or bowel inflammation (Radon et al., 2015; van Beelen Granlund et al., 2013). Expression levels often correlate with REG1B and other REG1 family proteins (van Beelen Granlund et al., 2013). |

This dataset was downloaded from Kaggle. However, the raw data is available directly from the authors through the published journal. Both files contain the same values, apart from slight changes in the headings, such as the availability of measurement units. For example: ‘Sample_ID’ in the original as opposed to ‘sample_id’ from Kaggle, and ‘REG1A (ng/ml)’ as opposed to ‘REG1A’. As it does not change any content in the cells, the dataset uploaded within the .zip file is the dataset downloaded from Kaggle.

### Data Preprocessing 
Explained in the Jupyter notebook. To summarize, the final features are Creatinine, LYVE1, REG1B, TFF1, age and sex. Although the focus is on urinary biomarkers, creatinine, age and sex would always be factors that are a part of clinical decision making, and thus can be included. 

### Model Development

When developing a classifier model that will distinguish between healthy controls, benign controls, and pancreatic cancer cases, the most important metric to consider is recall when predicting cases of pancreatic cancer (Hicks et al., 2022). In the clinical scenario, it is far more dangerous to produce a false negative than a false positive (Hicks et al., 2022). If a pancreatic cancer patient is wrongfully classified as healthy, the potential likelihood of detecting the cancer early decreases, further minimizing survival chances (Bartlett et al., 2021).

Distinguishing between non-pancreatic cancer-related cases and pancreatic cancer is one of the main aims of this classifier. Due to the symptomatic similarities between pancreatic cancer and other hepatobiliary diseases, it is important to accurately determine the difference between the two patient classes. Minimizing misdiagnosis and delays in appropriate treatment in both cases will be crucial for improving patient outcomes and reducing healthcare burden.

To develop a classification model that can achieve these aims, several methods were considered. Each model was first developed using a baseline model with basic parameters. Afterwhich a grid search was used to find the best hyperparameters, with scoring highest in recall.

#### Baseline Model: Logistic Regression
Logistic regression is a supervised learning algorithm that works as a discriminative classifier in both binary and multi-class classifications (Bisong, 2019). It determines the probability of a sample belonging to a particular class (diagnosis) by mapping and applying it to the log function (Kumar, 2023).

#### Advanced Model: Random Forest
Random Forest is a supervised learning algorithm that can be used in both classification and regression models (Shafi, 2024). It combines the outputs of multiple decision trees to generate a more accurate result (Te Beest et al., 2017). Random Forest is generally effective on high-dimensional datasets, as its many trees can break down the complex tabular structure of the data (Te Beest et al., 2017).

#### Advanced Model: XGBoost
XGBoost is a scalable machine-learning algorithm made for supervised learning tasks. Similar to Random Forest, it is an ensemble of decision trees (Tarwidi et al., 2023). For this dataset, XGBoost may have an advantage in its ability to focus on certain outcomes, such as distinguishing between the cancer class and the other hepatobiliary disease class (Chen & Guestrin, 2016). During hyperparameter tuning, the specific focus here was distinction between class 1 and class 2 (the patient classes). The initial assumption was that this would perform the best. 

### Model Comparison

To compare the models, classification reports and confusion matrices were generated for each of the models developed, with the most focus placed on recall in class 2 (pancreatic cancer prediction). Both Random Forest and XGBoost outperformed logistic regression in terms of accuracy by 1%. However, logistic regression outperformed both advanced models in class 2 recall by 10%. Based on this metric, the best model selected was the baseline model using logistic regression.

This differs from the initial assumption that XGBoost would yield the best results, given that it is the more complex and powerful algorithm. However, there are some possible reasons why logistic regression outperformed the more advanced models. Firstly, the comparatively small dataset could work better with simpler models like logistic regression. Meanwhile, more advanced models may have overfit the data, given the limited features. Additionally, the SHAP analysis showed a strong linear relationship between the variables in class 2, which could explain why logistic regression was able to perform better when making predictions for class 2.

### Explainable Analysis 

Feature importance, LIME, and SHAP analyses were conducted to understand how the model worked.

SHAP analysis indicated that all values, with the exception of creatinine and sex, had a strong linear relationship with increasing the SHAP values, showcasing their importance and relevance in determining a predictor.

LIME shows an approximation of how a data point was likely estimated. The output in the Jupyter Notebook displays the thresholds that determine the classes into which the prediction is likely to fall. In this case, the TFF1 and age values were particularly influential in the decision-making process, while creatinine and LYVE1 were less influential.

However, the feature importance graph showed that creatinine has the largest influence on the predictive process, while age is the least important. This makes clinical decision-making based on these features tougher, as it highlights the importance of a combination of different biomarkers to make an accurate assessment.

### Clinical Decision Making
While the model had a high recall, the contradictions within the explainable analysis suggest that the model needs further tuning. It is likely that a larger dataset is required to generate a more stable set of relationships to develop a solid classification model. Furthermore, a combined interpretation of multiple biomarkers is necessary for predicting pancreatic cancer.

### Limitations and Future Model Development

When building the model, factors like age, sex and creatinine were considered to be dropped. However, upon experimentation, these values reduced the models accuracy, suggesting some importance of these factors, especially age. This suggests that urinary biomarkers by itself may not be enough. Other biomarkers like the plasma CA19_9 could have potentially improved the models performance, and have better use in the clinical context.  

Future work can focus on refining the model using a larger database, along with additional features that are appropriate to a screening test. Additionally, developing a model that can also succeed in differentiating between stages of cancer by using slight differences between biomarker variations could make a significant clinical impact on patient outcomes.

### References
1. Asif, A. A., Hussain, H., & Chatterjee, T. (n.d.). Extraordinary Creatinine Level: A Case Report. Cureus, 12(7), e9076. https://doi.org/10.7759/cureus.9076
2. Bartlett, E. C., Silva, M., Callister, M. E., & Devaraj, A. (2021). False-Negative Results in Lung Cancer Screening—Evidence and Controversies. Journal of Thoracic Oncology, 16(6), 912–921. https://doi.org/10.1016/j.jtho.2021.01.160
3. Bisong, E. (2019). Logistic Regression. In E. Bisong (Ed.), Building Machine Learning and Deep Learning Models on Google Cloud Platform: A Comprehensive Guide for Beginners (pp. 243–250). Apress. https://doi.org/10.1007/978-1-4842-4470-8_
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.293985
5. Debernardi, S., O’Brien, H., Algahmdi, A. S., Malats, N., Stewart, G. D., Plješa-Ercegovac, M., Costello, E., Greenhalf, W., Saad, A., Roberts, R., Ney, A., Pereira, S. P., Kocher, H. M., Duffy, S., Blyuss, O., & Crnogorac-Jurcevic, T. (2020). A combination of urinary biomarker panel and PancRISK score for earlier detection of pancreatic cancer: A case-control study. PLoS Medicine, 17(12), e1003489. https://doi.org/10.1371/journal.pmed.100
6. 489
eBioMedicine. (2022). Emerging biomarkers for early diagnosis of pancreatic cancer. eBioMedicine, 79. https://doi.org/10.1016/j.ebiom.2022.1
7. 4064
Hicks, S. A., Strümke, I., Thambawita, V., Hammou, M., Riegler, M. A., Halvorsen, P., & Parasa, S. (2022). On evaluation metrics for medical applications of artificial intelligence. Scientific Reports, 12, 5979. https://doi.org/10.1038/s41598-022-0
8. 954-8
Kumar, S. (2023, November 14). 4. Assumptions and Limitations of Logistic Regression: Navigating the Nuances. Medium. https://medium.com/@skme20417/4-assumptions-and-limitations-of-logistic-regression-navigating-the-nuances-8ef24
9. cc7a01
McGuigan, A., Kelly, P., Turkington, R. C., Jones, C., Coleman, H. G., & McCain, R. S. (2018). Pancreatic cancer: A review of clinical diagnosis, epidemiology, treatment and outcomes. World Journal of Gastroenterology, 24(43), 4846–4861. https://doi.org/10.3748/wjg.v24.
10. 43.4846
Partyka, O., Pajewska, M., Kwaśniewska, D., Czerw, A., Deptała, A., Budzik, M., Cipora, E., Gąska, I., Gazdowicz, L., Mielnik, A., Sygit, K., Sygit, M., Krzych-Fałta, E., Schneider-Matyka, D., Grochans, S., Cybulska, A. M., Drobnik, J., Bandurska, E., Ciećko, W., … Kozłowski, R. (2023). Overview of Pancreatic Cancer Epidemiology in Europe and Recommendations for Screening in High-Risk Populations. Cancers, 15(14), 3634. https://doi.org/10.3390/cancer
11. 15143634
Radon, T. P., Massat, N. J., Jones, R., Alrawashdeh, W., Dumartin, L., Ennis, D., Duffy, S. W., Kocher, H. M., Pereira, S. P., Guarner posthumous, L., Murta-Nascimento, C., Real, F. X., Malats, N., Neoptolemos, J., Costello, E., Greenhalf, W., Lemoine, N. R., & Crnogorac-Jurcevic, T. (2015). Identification of a Three-Biomarker Panel in Urine for Early Detection of Pancreatic Adenocarcinoma. Clinical Cancer Research: An Official Journal of the American Association for Cancer Research, 21(15), 3512–3521. https://doi.org/10.1158/1078-0432.C
12. R-14-2467
Rawla, P., Sunkara, T., & Gaduputi, V. (2019). Epidemiology of Pancreatic Cancer: Global Trends, Etiology and Risk Factors. World Journal of Oncology, 10(1), 10–27. https://doi.org/10.147
13. 0/wjon1166
Samir, S., El-Ashry, M., Soliman, W., & Hassan, M. (2024). Urinary biomarkers analysis as a diagnostic tool for early detection of pancreatic adenocarcinoma: Molecular quantification approach. Computational Biology and Chemistry, 112, 108171. https://doi.org/10.1016/j.compbiolchem
14. 2024.108171
Shafi, A. (2024, October 1). Random Forest Classifier Tutorial: How to Build a Random Forest in Python. DataCamp. https://www.datacamp.com/tutorial/random-forests-clas
15. ifier-python
Tang, K. W. A., Toh, Q. C., & Teo, B. W. (2015). Normalisation of urinary biomarkers to creatinine for clinical practice and research – when and why. Singapore Medical Journal, 56(1), 7–10. https://doi.org/10.11622
16. smedj.2015003
Tarwidi, D., Pudjaprasetya, S. R., Adytia, D., & Apri, M. (2023). An optimized XGBoost-based machine learning method for predicting wave run-up on a sloping beach. MethodsX, 10, 102119. https://doi.org/10.1016/j.mex.2023.102119
Tatjana Crnogorac-Jurcevic. (n.d.). New tests for early detection of pancreatic cancer offer significant hope—Queen Mary University of London. Queen Mary University of London. Retrieved 30 April 2025, from https://www.qmul.ac.uk/research/featured-research/new-tests-for-early-detection-of-pancreatic-cancer-offer-s
17. gnificant-hope/
Te Beest, D. E., Mes, S. W., Wilting, S. M., Brakenhoff, R. H., & Van De Wiel, M. A. (2017). Improved high-dimensional prediction with Random Forests by the use of co-data. BMC Bioinformatics, 18(1), 584. https://doi.org/10.1186/
18. 12859-017-1993-1
van Beelen Granlund, A., Østvik, A. E., Brenna, Ø., Torp, S. H., Gustafsson, B. I., & Sandvik, A. K. (2013). REG gene expression in inflamed and healthy colon mucosa explored by in situ hybridisation. Cell and Tissue Research, 352(3), 639–646. https://doi.org/10.100700441-013-1592-z

 



```python

```
