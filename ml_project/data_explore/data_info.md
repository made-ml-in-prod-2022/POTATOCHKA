### General view on data
|    |   age |   sex |   cp |   trestbps |   chol |   fbs |   restecg |   thalach |   exang |   oldpeak |   slope |   ca |   thal |   condition |
|---:|------:|------:|-----:|-----------:|-------:|------:|----------:|----------:|--------:|----------:|--------:|-----:|-------:|------------:|
|  0 |    69 |     1 |    0 |        160 |    234 |     1 |         2 |       131 |       0 |       0.1 |       1 |    1 |      0 |           0 |
|  1 |    69 |     0 |    0 |        140 |    239 |     0 |         0 |       151 |       0 |       1.8 |       0 |    2 |      0 |           0 |
|  2 |    66 |     0 |    0 |        150 |    226 |     0 |         0 |       114 |       0 |       2.6 |       2 |    0 |      0 |           0 |
|  3 |    65 |     1 |    0 |        138 |    282 |     1 |         2 |       174 |       0 |       1.4 |       1 |    1 |      0 |           1 |
|  4 |    64 |     1 |    0 |        110 |    211 |     0 |         2 |       144 |       1 |       1.8 |       1 |    0 |      0 |           0 |
### Feats description
There are 13 attributes<br><br>age: age in years<br>sex: sex (1 = male; 0 = female)<br>cp: chest pain type<br>-- Value 0: typical angina<br>-- Value 1: atypical angina<br>-- Value 2: non-anginal pain<br>-- Value 3: asymptomatic<br>trestbps: resting blood pressure (in mm Hg on admission to the hospital)<br>chol: serum cholestoral in mg/dl<br>fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)<br>restecg: resting electrocardiographic results<br>-- Value 0: normal<br>-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)<br>-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria<br>thalach: maximum heart rate achieved<br>exang: exercise induced angina (1 = yes; 0 = no)<br>oldpeak = ST depression induced by exercise relative to rest<br>slope: the slope of the peak exercise ST segment<br>-- Value 0: upsloping<br>-- Value 1: flat<br>-- Value 2: downsloping<br>ca: number of major vessels (0-3) colored by flourosopy<br>thal: 0 = normal; 1 = fixed defect; 2 = reversable defect<br>and the label<br>condition: 0 = no disease, 1 = disease
### Feats in dataset
binary feats = sex, fbs, exang, condition<br>categorical feats = cp, restecg, slope, ca, thal<br>numerical feats = age, trestbps, chol, thalach, oldpeak<br>
### Data shape
rows = 297, cols = 14
### Data types
|           | 0       |
|:----------|:--------|
| age       | int64   |
| sex       | int64   |
| cp        | int64   |
| trestbps  | int64   |
| chol      | int64   |
| fbs       | int64   |
| restecg   | int64   |
| thalach   | int64   |
| exang     | int64   |
| oldpeak   | float64 |
| slope     | int64   |
| ca        | int64   |
| thal      | int64   |
| condition | int64   |
### Nans in data
|           |   0 |
|:----------|----:|
| age       |   0 |
| sex       |   0 |
| cp        |   0 |
| trestbps  |   0 |
| chol      |   0 |
| fbs       |   0 |
| restecg   |   0 |
| thalach   |   0 |
| exang     |   0 |
| oldpeak   |   0 |
| slope     |   0 |
| ca        |   0 |
| thal      |   0 |
| condition |   0 |
### Data statistics
|       |       age |        sex |         cp |   trestbps |     chol |        fbs |    restecg |   thalach |      exang |   oldpeak |      slope |         ca |       thal |   condition |
|:------|----------:|-----------:|-----------:|-----------:|---------:|-----------:|-----------:|----------:|-----------:|----------:|-----------:|-----------:|-----------:|------------:|
| count | 297       | 297        | 297        |   297      | 297      | 297        | 297        |  297      | 297        | 297       | 297        | 297        | 297        |  297        |
| mean  |  54.5421  |   0.676768 |   2.15825  |   131.694  | 247.35   |   0.144781 |   0.996633 |  149.599  |   0.326599 |   1.05556 |   0.602694 |   0.676768 |   0.835017 |    0.461279 |
| std   |   9.04974 |   0.4685   |   0.964859 |    17.7628 |  51.9976 |   0.352474 |   0.994914 |   22.9416 |   0.469761 |   1.16612 |   0.618187 |   0.938965 |   0.95669  |    0.49934  |
| min   |  29       |   0        |   0        |    94      | 126      |   0        |   0        |   71      |   0        |   0       |   0        |   0        |   0        |    0        |
| 25%   |  48       |   0        |   2        |   120      | 211      |   0        |   0        |  133      |   0        |   0       |   0        |   0        |   0        |    0        |
| 50%   |  56       |   1        |   2        |   130      | 243      |   0        |   1        |  153      |   0        |   0.8     |   1        |   0        |   0        |    0        |
| 75%   |  61       |   1        |   3        |   140      | 276      |   0        |   2        |  166      |   1        |   1.6     |   1        |   1        |   2        |    1        |
| max   |  77       |   1        |   3        |   200      | 564      |   1        |   2        |  202      |   1        |   6.2     |   2        |   3        |   2        |    1        |
### Data correlation
|           |        age |         sex |          cp |   trestbps |        chol |          fbs |    restecg |     thalach |        exang |     oldpeak |       slope |         ca |       thal |   condition |
|:----------|-----------:|------------:|------------:|-----------:|------------:|-------------:|-----------:|------------:|-------------:|------------:|------------:|-----------:|-----------:|------------:|
| age       |  1         | -0.0923995  |  0.110471   |  0.290476  |  0.202644   |  0.132062    |  0.149917  | -0.394563   |  0.0964888   |  0.197123   |  0.159405   |  0.36221   |  0.120795  |  0.227075   |
| sex       | -0.0923995 |  1          |  0.00890803 | -0.0663402 | -0.198089   |  0.0388503   |  0.0338968 | -0.060496   |  0.143581    |  0.106567   |  0.033345   |  0.0919248 |  0.370556  |  0.278467   |
| cp        |  0.110471  |  0.00890803 |  1          | -0.0369797 |  0.0720883  | -0.0576631   |  0.0639047 | -0.339308   |  0.377525    |  0.203244   |  0.151079   |  0.235644  |  0.266275  |  0.408945   |
| trestbps  |  0.290476  | -0.0663402  | -0.0369797  |  1         |  0.131536   |  0.18086     |  0.149242  | -0.0491077  |  0.0666911   |  0.191243   |  0.121172   |  0.0979538 |  0.130612  |  0.15349    |
| chol      |  0.202644  | -0.198089   |  0.0720883  |  0.131536  |  1          |  0.0127083   |  0.165046  | -7.4568e-05 |  0.0593389   |  0.0385958  | -0.00921524 |  0.115945  |  0.0234408 |  0.0802848  |
| fbs       |  0.132062  |  0.0388503  | -0.0576631  |  0.18086   |  0.0127083  |  1           |  0.0688311 | -0.00784236 | -0.000893082 |  0.00831067 |  0.047819   |  0.152086  |  0.0510379 |  0.00316683 |
| restecg   |  0.149917  |  0.0338968  |  0.0639047  |  0.149242  |  0.165046   |  0.0688311   |  1         | -0.0722896  |  0.0818739   |  0.113726   |  0.135141   |  0.129021  |  0.0136119 |  0.166343   |
| thalach   | -0.394563  | -0.060496   | -0.339308   | -0.0491077 | -7.4568e-05 | -0.00784236  | -0.0722896 |  1          | -0.384368    | -0.34764    | -0.389307   | -0.268727  | -0.258386  | -0.423817   |
| exang     |  0.0964888 |  0.143581   |  0.377525   |  0.0666911 |  0.0593389  | -0.000893082 |  0.0818739 | -0.384368   |  1           |  0.28931    |  0.250572   |  0.148232  |  0.323268  |  0.421355   |
| oldpeak   |  0.197123  |  0.106567   |  0.203244   |  0.191243  |  0.0385958  |  0.00831067  |  0.113726  | -0.34764    |  0.28931     |  1          |  0.579037   |  0.294452  |  0.336809  |  0.424052   |
| slope     |  0.159405  |  0.033345   |  0.151079   |  0.121172  | -0.00921524 |  0.047819    |  0.135141  | -0.389307   |  0.250572    |  0.579037   |  1          |  0.109761  |  0.260096  |  0.333049   |
| ca        |  0.36221   |  0.0919248  |  0.235644   |  0.0979538 |  0.115945   |  0.152086    |  0.129021  | -0.268727   |  0.148232    |  0.294452   |  0.109761   |  1         |  0.248825  |  0.463189   |
| thal      |  0.120795  |  0.370556   |  0.266275   |  0.130612  |  0.0234408  |  0.0510379   |  0.0136119 | -0.258386   |  0.323268    |  0.336809   |  0.260096   |  0.248825  |  1         |  0.520516   |
| condition |  0.227075  |  0.278467   |  0.408945   |  0.15349   |  0.0802848  |  0.00316683  |  0.166343  | -0.423817   |  0.421355    |  0.424052   |  0.333049   |  0.463189  |  0.520516  |  1          |
## Histograms
![histogram](hist.png)