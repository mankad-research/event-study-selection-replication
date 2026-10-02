import pandas as pd
import numpy as np

df = pd.read_stata('/mnt/user-data/uploads/MediaDiscretionDataWithNAMES.dta', convert_categoricals=False)

df['returns'] = df['WSJ_abret_10']
covered = (df['WSJCoverage1Day']==1) | (df['WSJCoverage']==1)
uncovered = (df['WSJCoverage1Day']==0) & (df['WSJCoverage']==0)
df.loc[uncovered, 'returns'] = df.loc[uncovered, 'abret_10']

keep_cols = ['returns','MarketValue','PRAnnouncementDate','Female','MediaCitations','Outsider','Analyst']
df2 = df.dropna(subset=keep_cols).copy()
df2['C'] = covered.loc[df2.index].astype(int)

df2['logMV'] = np.log(df2['MarketValue'])
# Analyst is already log(1+n) in the source data (verified: exp(Analyst)-1 gives integers 0,1,2,3,...)
df2['posReturns'] = np.maximum(df2['returns'], 0.0)
df2['negReturns'] = np.maximum(-df2['returns'], 0.0)

COVAR_COLS = ['Female','MediaCitations','logMV','Outsider','Analyst']

df2.to_pickle('working_data.pkl')
print("n total:", len(df2))
print("n covered (original/media sample):", df2['C'].sum())
print("n uncovered (non-discretionary pool):", (1-df2['C']).sum())
print("coverage rate:", df2['C'].mean())
print("returns describe:\n", df2['returns'].describe())
