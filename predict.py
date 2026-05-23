import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import json
import glob
import os

INPUT_PATH = '/data/input/holdout.csv'
OUTPUT_DIR = '/data/output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'predictions.csv')
MODELS_DIR = 'models'

def add_features_inference(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    df['DIST_TO_ADM_CENTER'] = np.log1p(df['DIST_TO_ADM_CENTER'].fillna(0))
    df['TOTAL_RANK_COMPS_CANNIBALS'] = df['TOTAL_RANK_COMPS_CANNIBALS'].fillna(0).astype(np.int64)
    
    df['families_per_competitor'] = df['HuffFamilies'] / (df['TOTAL_RANK_COMPS_CANNIBALS'] + 1)
    df['attraction_per_square'] = df['TRADING_PATCH_SCORE'] / (df['TRADE_SQUARE'] + 1e-6)
    df['families_x_24h'] = df['HuffFamilies'] * df['ENTIRE_DAY']
    df['traffic_x_24h'] = df['ROUTES_CNT'] * df['ENTIRE_DAY']
    df['families_x_attraction'] = df['HuffFamilies'] * df['TRADING_PATCH_SCORE']
    return df

def main():
    print("Starting inference...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found.")
        return

    with open(f'{MODELS_DIR}/metadata.json', 'r') as f:
        meta = json.load(f)
    features = meta['features']

    raw_df = pd.read_csv(INPUT_PATH)
    df_proc = add_features_inference(raw_df)

    model_files = glob.glob(f'{MODELS_DIR}/*.cbm')
    if not model_files:
        print("Error: No models found in models/ directory.")
        return

    print(f"Loaded {len(model_files)} models. Processing...")
    
    ensemble_preds = []
    for model_path in model_files:
        model = CatBoostRegressor()
        model.load_model(model_path)
        
        preds_log = model.predict(df_proc[features])
        preds_real = np.maximum(np.expm1(preds_log), 0)
        ensemble_preds.append(preds_real)
        
    final_preds = np.mean(ensemble_preds, axis=0)
    
    result = pd.DataFrame({
        'id': raw_df['ID'].values,
        'PREDICT': final_preds
    })
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()