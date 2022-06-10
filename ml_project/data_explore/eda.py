import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./data/raw/heart_cleveland_upload.csv')

description = """There are 13 attributes

age: age in years
sex: sex (1 = male; 0 = female)
cp: chest pain type
-- Value 0: typical angina
-- Value 1: atypical angina
-- Value 2: non-anginal pain
-- Value 3: asymptomatic
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
-- Value 0: upsloping
-- Value 1: flat
-- Value 2: downsloping
ca: number of major vessels (0-3) colored by flourosopy
thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
and the label
condition: 0 = no disease, 1 = disease"""

binary_feats = list(data.nunique()[data.nunique() == 2].index)
categorical_feats = list(data.nunique()[(data.nunique() > 2) & (data.nunique() < 10)].index)
numerical_feats = list(data.nunique()[data.nunique() > 10].index)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
data.hist(ax=ax)
fig.tight_layout()
histogram = "hist.png"
fig.savefig(f"./data_explore/{histogram}")

with open('data_explore/data_info.md', 'w') as f:
    f.write('### General view on data\n')
    f.write(data.head().to_markdown())
    f.write('\n')
    f.write('### Feats description\n')
    f.write('<br>'.join(description.split('\n')))
    f.write('\n')
    f.write('### Feats in dataset\n')
    f.write('binary feats = ' + ', '.join(binary_feats) + '<br>')
    f.write('categorical feats = ' + ', '.join(categorical_feats) + '<br>')
    f.write('numerical feats = ' + ', '.join(numerical_feats) + '<br>')
    f.write('\n')
    f.write('### Data shape\n')
    f.write(f'rows = {data.shape[0]}, cols = {data.shape[1]}')
    f.write('\n')
    f.write('### Data types\n')
    f.write(pd.DataFrame(data.dtypes).to_markdown())
    f.write('\n')
    f.write('### Nans in data\n')
    f.write(pd.DataFrame(data.isna().sum()).to_markdown())
    f.write('\n')
    f.write('### Data statistics\n')
    f.write(pd.DataFrame(data.describe()).to_markdown())
    f.write('\n')
    f.write('### Data correlation\n')
    f.write(pd.DataFrame(data.corr()).to_markdown())
    f.write('\n')
    f.write("## Histograms\n")
    f.write(f"![histogram]({histogram})")
