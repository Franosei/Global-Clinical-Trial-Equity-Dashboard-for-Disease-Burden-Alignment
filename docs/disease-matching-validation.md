# Disease matching validation

**Status: INCOMPLETE.** Generated 2026-11-22T21:56:36+00:00. 
The status becomes complete only when every disease has enough usable labels, precision and recall are estimable, reviewers are named, and the project owner records that the review was independently verified.

## Method

ClinicalTrials.gov interventional records are retrieved through the broad, versioned queries in `config/disease_review_queries.yaml`. Records are stratified by the production classifier prediction, and a seeded random sample is taken from each stratum. Reviewers see the NCT ID, title, registered conditions and target definition, but not the prediction. `include` and `exclude` labels are scored; blank and `uncertain` labels are not.

Precision and recall use inverse-sampling weights because classifier-negative near-matches are deliberately over-sampled. The estimates describe the published ClinicalTrials.gov search frame, not all possible registry terminology. Contract tests and records used to tune the rules must not be added to the blinded sample.

## Results by disease

| Disease | Reviewed | Near matches | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| HIV/AIDS | 0 | 0 | — | — |
| Tuberculosis | 0 | 0 | — | — |
| Malaria | 0 | 0 | — | — |
| Lower respiratory infections | 0 | 0 | — | — |
| Diarrhoeal diseases | 0 | 0 | — | — |
| Viral hepatitis | 0 | 0 | — | — |
| Meningitis | 0 | 0 | — | — |
| Measles | 0 | 0 | — | — |
| Neglected tropical diseases | 0 | 0 | — | — |
| Maternal conditions | 0 | 0 | — | — |
| Neonatal conditions | 0 | 0 | — | — |
| Trachea, bronchus and lung cancers | 0 | 0 | — | — |
| Breast cancer | 0 | 0 | — | — |
| Cervical cancer | 0 | 0 | — | — |
| Colorectal cancer | 0 | 0 | — | — |
| Liver cancer | 0 | 0 | — | — |
| Stomach cancer | 0 | 0 | — | — |
| Prostate cancer | 0 | 0 | — | — |
| Ischaemic heart disease | 0 | 0 | — | — |
| Stroke | 0 | 0 | — | — |
| Hypertensive heart disease | 0 | 0 | — | — |
| Diabetes mellitus | 0 | 0 | — | — |
| Kidney diseases | 0 | 0 | — | — |
| Chronic obstructive pulmonary disease | 0 | 0 | — | — |
| Asthma | 0 | 0 | — | — |
| Depressive disorders | 0 | 0 | — | — |
| Anxiety disorders | 0 | 0 | — | — |
| Alzheimer's disease and dementias | 0 | 0 | — | — |
| Cirrhosis of the liver | 0 | 0 | — | — |
| Road traffic injuries | 0 | 0 | — | — |
| Self-harm | 0 | 0 | — | — |
| Sickle cell disease | 0 | 0 | — | — |
| Congenital anomalies | 0 | 0 | — | — |

## Published matching rules

The machine-readable source is `data/validation/disease_matching/matching_rules.csv`. `exclude phrase` removes only that phrase before matching; `exclude study if` rejects the whole study for that disease.

| Disease | Include terms | Excluded phrases | Exclude whole study if |
| --- | --- | --- | --- |
| HIV/AIDS | hiv; acquired immunodeficiency; aids related; aids defining | — | — |
| Tuberculosis | tuberculosis; tb; mycobacterium tuberculosis | — | — |
| Malaria | malaria; plasmodium | — | — |
| Lower respiratory infections | pneumonia; lower respiratory tract infection; lower respiratory infection; bronchiolitis; acute lower respiratory; lri | — | covid; sars cov 2; coronavirus disease 2019; 2019 ncov |
| Diarrhoeal diseases | diarrhoea; diarrhea; diarrhoeal; diarrheal; gastroenteritis; rotavirus; cholera; shigella; shigellosis | chemotherapy induced diarrhea; chemotherapy induced diarrhoea; irritable bowel syndrome with diarrhea | — |
| Viral hepatitis | hepatitis b; hepatitis c; viral hepatitis; hepatitis a; hepatitis e; hepatitis d; hepatitis delta; hbv; hcv; hev | — | — |
| Meningitis | meningitis; meningococcal; meningococcus; neisseria meningitidis | — | — |
| Measles | measles; rubeola; morbillivirus | — | — |
| Neglected tropical diseases | dengue; leishmaniasis; schistosomiasis; chagas; trypanosomiasis; lymphatic filariasis; onchocerciasis; trachoma; rabies; leprosy; hookworm; soil transmitted helminth; cysticercosis; echinococcosis; ascariasis; trichuriasis; fascioliasis; clonorchiasis; opisthorchiasis; sleeping sickness; bilharzia; river blindness; hydatid; filariasis; hansen disease; helminth | — | — |
| Maternal conditions | preeclampsia; pre eclampsia; eclampsia; postpartum hemorrhage; postpartum haemorrhage; gestational hypertension; miscarriage; ectopic pregnancy; hyperemesis gravidarum; maternal sepsis; puerperal; post partum hemorrhage; pregnancy induced hypertension; obstructed labor; obstructed labour; maternal mortality; maternal death | — | — |
| Neonatal conditions | neonatal; newborn; preterm infant; preterm birth; prematurity; birth asphyxia; neonatal sepsis; bronchopulmonary dysplasia; neonatal encephalopathy; hypoxic ischemic encephalopathy | neonatal diabetes | — |
| Trachea, bronchus and lung cancers | lung cancer; non small cell lung; small cell lung; nsclc; lung carcinoma; lung neoplasm; lung adenocarcinoma; sclc; bronchogenic carcinoma; lung tumor; lung tumour | — | — |
| Breast cancer | breast cancer; breast carcinoma; breast neoplasm; breast tumor; breast tumour; invasive lobular carcinoma | — | — |
| Cervical cancer | cervical cancer; cervix cancer; cervical carcinoma; cervical neoplasm; uterine cervical neoplasm; cervix carcinoma | — | — |
| Colorectal cancer | colorectal cancer; colon cancer; rectal cancer; colorectal carcinoma; colorectal neoplasm; colon carcinoma; rectal carcinoma; colonic neoplasm; rectal neoplasm; crc | — | — |
| Liver cancer | hepatocellular carcinoma; liver cancer; hcc; hepatocellular cancer; liver neoplasm; hepatoma; liver carcinoma | — | — |
| Stomach cancer | gastric cancer; stomach cancer; gastric carcinoma; gastric adenocarcinoma; stomach neoplasm; gastric neoplasm | — | — |
| Prostate cancer | prostate cancer; prostatic cancer; prostate carcinoma; prostatic neoplasm; prostate adenocarcinoma; prostatic adenocarcinoma; crpc; mcrpc | — | — |
| Ischaemic heart disease | coronary artery disease; myocardial infarction; acute coronary syndrome; ischemic heart; ischaemic heart; coronary heart disease; coronary disease; angina pectoris; myocardial ischemia; myocardial ischaemia; ihd | — | — |
| Stroke | stroke; cerebrovascular; cerebral infarction; intracerebral hemorrhage; intracerebral haemorrhage; subarachnoid hemorrhage; subarachnoid haemorrhage; cbvd | heat stroke; heatstroke | — |
| Hypertensive heart disease | hypertensive heart disease; hypertensive heart failure; hypertensive cardiomyopathy; hypertensive cardiac disease | pulmonary hypertension; pulmonary arterial hypertension; portal hypertension; intracranial hypertension; ocular hypertension; gestational hypertension; pregnancy induced hypertension; intra abdominal hypertension | — |
| Diabetes mellitus | diabetes; diabetic; t2dm; t1dm; t2d; t1d; dm | diabetes insipidus | — |
| Kidney diseases | chronic kidney; kidney disease; kidney failure; renal failure; renal insufficiency; end stage renal; end stage kidney; diabetic nephropathy; glomerulonephritis; esrd; eskd; ckd; dialysis | — | — |
| Chronic obstructive pulmonary disease | copd; chronic obstructive pulmonary; chronic obstructive lung; emphysema; chronic bronchitis | — | — |
| Asthma | asthma; asthmatic | — | — |
| Depressive disorders | depression; depressive; major depressive; dysthymia; dysthymic | respiratory depression; bone marrow depression; marrow depression; myocardial depression; cortical spreading depression; immune depression; st segment depression | — |
| Anxiety disorders | anxiety disorder; generalized anxiety; generalised anxiety; panic disorder; social anxiety; agoraphobia; phobia; phobic disorder; separation anxiety | — | — |
| Alzheimer's disease and dementias | alzheimer; dementia | — | — |
| Cirrhosis of the liver | cirrhosis; liver fibrosis; hepatic fibrosis; chronic liver disease | — | — |
| Road traffic injuries | road traffic; traffic accident; traffic injury; motor vehicle; road injury; transport injury; car crash; vehicle collision | — | — |
| Self-harm | suicide; suicidal; self harm; self injury; self injurious; parasuicide | — | — |
| Sickle cell disease | sickle cell; drepanocytosis; hbss; hemoglobin ss; haemoglobin ss | — | — |
| Congenital anomalies | congenital heart; congenital anomal; congenital malformation; birth defect; neural tube defect; spina bifida; cleft lip; cleft palate; down syndrome; trisomy 21; tetralogy of fallot; hypoplastic left heart; gastroschisis; congenital diaphragmatic hernia | — | — |

## Reproduction

```powershell
py -m py.disease_validation sample --per-stratum 20
# Complete review_sample.csv without opening review_predictions.csv.
py -m py.disease_validation score --min-reviewed-per-disease 10
```
