def sanitize_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[', '').str.replace(')', '')
    return df