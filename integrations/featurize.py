import numpy as np
import pandas as pd


def featurize_df(df):
    df = df.copy()
    for col in ['age', 'balance', 'pdays', 'previous', 'duration', 'campaign']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 120], labels=['<25','25-34','35-44','45-54','55-64','>=65'])
        df['age_sq'] = df['age'] ** 2
        df['is_senior'] = (df['age'] >= 60).astype(int)
    
    if 'balance' in df.columns:
        df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
        df['balance_pos'] = (df['balance'] > 0).astype(int)
        df['balance_per_age'] = df['balance'] / (df['age'].replace(0, np.nan))
        df['balance_per_age'] = df['balance_per_age'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if 'pdays' in df.columns:
        df['pdays_flag'] = (df['pdays'] == -1).astype(int)
        df['pdays_log'] = np.log1p(df['pdays'].clip(lower=0))
        df['pdays_capped'] = df['pdays'].where(df['pdays'] >= 0, 1000)
    
    if 'previous' in df.columns:
        df['previous_flag'] = (df['previous'] > 0).astype(int)
        df['previous_log'] = np.log1p(df['previous'])
    
    if 'duration' in df.columns:
        df['duration_bin'] = pd.cut(df['duration'], bins=[-1, 60, 180, 600, 10000], labels=['short','mid','long','very_long']).astype(object)
        df['duration_log'] = np.log1p(df['duration'].clip(lower=0))
    
    if 'campaign' in df.columns:
        df['campaign_bin'] = pd.cut(df['campaign'], bins=[-1,0,1,2,3,5,100], labels=['0','1','2','3','4-5','6+']).astype(object)
        df['campaign_log'] = np.log1p(df['campaign'])
        df['campaign_ge_3'] = (df['campaign'] >= 3).astype(int)
    
    if 'campaign' in df.columns and 'previous' in df.columns:
        df['camp_prev'] = df['campaign'] * df['previous']
        df['prev_per_campaign'] = df['previous'] / (df['campaign'].replace(0, np.nan))
        df['prev_per_campaign'] = df['prev_per_campaign'].fillna(0)
    
    if 'month' in df.columns:
        try:
            months = {m.lower(): i for i, m in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], start=1)}
            df['month_num'] = df['month'].astype(str).str.lower().map(months).fillna(0).astype(int)
            # Meses pico observados en la distribución (mapeo fijo)
            peak = set(['mar','sep','oct','dec'])
            df['month_peak'] = df['month'].astype(str).str.lower().isin(peak).astype(int)
        except:
            df['month_num'] = 0
    
    if 'day' in df.columns:
        df['day_bin'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=['early','mid','late']).astype(object)
        df['is_early_month'] = (df['day'] <= 10).astype(int)

    # Ayudas categóricas simples (mapeos estáticos; sin fugas)
    if 'contact' in df.columns:
        df['contact'] = df['contact'].astype(str)
        df['is_cellular'] = (df['contact'].str.lower() == 'cellular').astype(int)
        df['is_telephone'] = (df['contact'].str.lower() == 'telephone').astype(int)

    if 'poutcome' in df.columns:
        df['poutcome'] = df['poutcome'].astype(str)
        df['pout_success'] = (df['poutcome'].str.lower() == 'success').astype(int)
        df['pout_failure'] = (df['poutcome'].str.lower() == 'failure').astype(int)

    if 'housing' in df.columns and 'loan' in df.columns:
        df['any_loan'] = ((df['housing'].astype(str).str.lower() == 'yes') | (df['loan'].astype(str).str.lower() == 'yes')).astype(int)

    if 'job' in df.columns:
        jb = df['job'].astype(str).str.lower()
        df['job_student'] = (jb == 'student').astype(int)
        df['job_retired'] = (jb == 'retired').astype(int)
        df['job_unemployed'] = (jb == 'unemployed').astype(int)

    if 'education' in df.columns:
        ed = df['education'].astype(str).str.lower()
        df['edu_tertiary'] = (ed == 'tertiary').astype(int)
        df['edu_secondary'] = (ed == 'secondary').astype(int)
        df['edu_primary'] = (ed == 'primary').astype(int)
    
    return df
