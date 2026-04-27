import requests
import pandas as pd
import numpy as np

for i in range(1, 3):
    # Create random dataset
    df = pd.DataFrame({
        'datetime': pd.date_range('2025-01-01', periods=1000, freq='h'),
        'energy_kwh': np.random.uniform(i*10, i*20, 1000) # Different scale for each
    })
    filename = f"test_db_{i}.csv"
    df.to_csv(filename, index=False)
    
    # Upload
    with open(filename, 'rb') as f:
        upload_res = requests.post("http://localhost:5000/upload", files={'file': f})
        print(f"Dataset {i} upload:", upload_res.json())
        
    # Train
    train_res = requests.post("http://localhost:5000/api/train")
    metrics = train_res.json()
    metrics_rmse = metrics['metrics']['linear_regression']['RMSE']
    print(f"Dataset {i} RMSE:", metrics_rmse)
