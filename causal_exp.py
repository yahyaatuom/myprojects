#Causal Experiment using DoWhy Library
#Q) does road proximity CAUSE deforestation?
import practice as np 
import pandas as pd

np.random.seed(42)
n = 500

elevation = np.random.normal(500, 200, n)

road_proximity = (elevation < 400).astype(int) + np.random.binomial(1, 0.3, n)
road_proximity = (road_proximity >=1).astype(int)

#Outcome
deforestation = (
    0.6 * road_proximity           # causal eefect of roads
    - 0.003 * elevation            # effect of elevation
    + np.random.normal(0, 0.1, n)  # noise
)
deforestation = np.clip(deforestation, 0, 1)

df = pd.DataFrame({
    'road_proximity': road_proximity,
    'elevation': elevation,
    'deforestation': deforestation
})

# Naive correlation
naive = df.groupby('road_proximity')['deforestation'].mean()
print('Naive group means (biased estimate):')
print(naive.round(3))
print(f'Naive effect: {(naive[1] - naive [0]):.3f}')

print('\nDataset sample:')
print(df.head(5).round(3))
print('\n(Next step: use DoWhy to estimate the TRUE causal effect)')