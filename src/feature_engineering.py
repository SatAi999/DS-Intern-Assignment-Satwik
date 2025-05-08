import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_interaction_terms(self, df, features):
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction_df = pd.DataFrame(poly.fit_transform(df[features]), 
                                      columns=poly.get_feature_names_out(features))
        interaction_df.drop(columns=features, inplace=True)  # drop original columns to avoid duplication
        df = pd.concat([df.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)
        return df

    def engineer_features(self, df):
        df['humidity_temp_ratio_zone1'] = df['zone1_humidity'] / (df['zone1_temperature'] + 1e-5)
        df['dew_pressure_interaction'] = df['dew_point'] * df['atmospheric_pressure']
        df['temp_diff_zone1_zone5'] = df['zone1_temperature'] - df['zone5_temperature']
        df['is_high_energy'] = (df['equipment_energy_consumption'] > df['equipment_energy_consumption'].median()).astype(int)
        return df

    def scale_numeric(self, df, numeric_cols):
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
