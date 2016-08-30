import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm
loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[:3]))
plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10, 10), diagonal='hist')
plt.show()
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']
y=np.matrix(intrate).transpose()
x1=np.matrix(fico).transpose()
x2=np.matrix(loanamt).transpose()
x=np.column_stack([x1,x2])
X=sm.add_constant(x)
model=sm.OLS(y,x)
f=model.fit()
f.summary()
