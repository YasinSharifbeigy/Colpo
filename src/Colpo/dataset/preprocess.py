
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Imputation:
    def __init__(self):
        # Learned statistics
        self.age_median = None
        self.age_scaler = None

    # ==============================================================
    #                       FIT
    # ==============================================================
    def fit(self, df):
        """
        Learn statistics from the training dataset only.
        """
        # ---- Fit Age median ----
        self.age_median = df["Age"].median()

        # ---- Fit StandardScaler using a DataFrame (keeps feature names) ----
        age_filled = df[["Age"]].fillna(self.age_median)
        self.age_scaler = StandardScaler().fit(age_filled)

        return self


    PAP_MAP = {
        "NILM": 0,
        "ENDOMETRII-CELL":0.75,
        "ASCUS": 1,
        "LSIL": 2,
        "ASCH": 3,
        "HSIL": 4,
        "SCC": 4.5,
        "AGC": 5,
        "AGCNOS": 5.5,
        "AGC-NEOPLASI": 6,
        "ADENO-CARCINOMA": 6.5,
        "UNSATISFACTORY": 7}

    # ==============================================================
    #                   TRANSFORM HELPERS
    # ==============================================================

    # ---------- HPV ENCODING ----------
    @staticmethod
    def encode_hpv(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0     # Missing → neutral
        x = str(x).strip().lower()
        if x == "No":
            return -2
        if x in ["Others"]:
            return -1
        if x in ["18", "hpv18"]:
            return 1
        if x in ["16", "hpv16"]:
            return 2
        return -1  # Unknown → treat as low risk

    def encode_hpv_from_columns(self, row):
        """
        row: a pandas Series with columns:
            '16', '18', 'Others'
        """
        # If all three are missing -> neutral (0)
        if row[["16", "18", "Others", "No"]].isna().all():
            return 0

        has_16 = bool(row.get("16", 0))
        has_18 = bool(row.get("18", 0))
        has_oth = bool(row.get("Others", 0))
        no_hpv = bool(row.get("No", 0))

        # No HPV at all (all zeros) -> -2
        if not has_16 and not has_18 and not has_oth:
            return -2

        codes = []
        if has_16:
            return 2.5
        if has_18:
            return 2
        if has_oth:
            return 1


    # ---------- PAP SMEAR ----------
    @staticmethod
    def encode_pap_category(x):
        if pd.isna(x) or str(x).strip() == "":
            return np.nan

        key = str(x)
        if key in Imputation.PAP_MAP:
            return Imputation.PAP_MAP[key]

        # Unknown category -> conservative choice (ASCUS)
        return 1

    # ---------- LUGOL ----------
    @staticmethod
    def encode_lugol_category(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0
        x = str(x).strip().lower()
        if x == "negative":
            return -1
        if x == "positive":
            return 1
        return 0  # unknown → treat as negative

    # ==============================================================
    #                     TRANSFORM
    # ==============================================================
    def transform(self, df):
        """
        Apply imputations + encodings to any dataframe.
        Requires .fit() to have been called first.
        Returns a NEW dataframe (does not modify original df).
        """
        df = df.copy()

        # ========================================================
        # AGE
        # ========================================================
        df["Age"] = df["Age"].fillna(self.age_median)
        df["Age"] = self.age_scaler.transform(df[["Age"]])

        # ========================================================
        # HPV ENCODING
        # ========================================================
        df["HPV_encoded"] = df.apply(self.encode_hpv_from_columns, axis=1)


        # ========================================================
        # PAP SMEAR
        # ========================================================
        df["Pap_level_raw"] = df["Pop"].apply(self.encode_pap_category)
        df["Pap_missing"] = df["Pap_level_raw"].isna().astype(int)
        df["Pap_level"] = df["Pap_level_raw"].fillna(0)
        df.drop(columns=["Pap_level_raw"], inplace=True)

        # ========================================================
        # LUGOL
        # ========================================================
        df["Lugol"] = df["Logul"].apply(self.encode_lugol_category)
        # df["Lugol_missing"] = df["Lugol_raw"].isna().astype(int)
        # df["Lugol_value"] = df["Lugol_raw"].fillna(0)
        # df.drop(columns=["Lugol_raw"], inplace=True)

        # ========================================================
        # Final selected columns (tabular features)
        # ========================================================
        final_cols = [
            "Age",
            "HPV_encoded",
            "Pap_level",
            "Pap_missing",
            "Lugol",
        ]

        return df[final_cols].astype(np.float32)

    # ==============================================================
    #                FIT + TRANSFORM CONVENIENCE
    # ==============================================================
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def prepare_logul(data: pd.DataFrame):
    data.loc[data['Logul'] == 'Negetive', 'Logul'] = 'Negative'
    data.loc[data['Logul'] == 'POSITIVE', 'Logul'] = 'Positive'
    return data


def prepare_hpv(df, HPV_col = ['16', '18', 'Others', "No"]):
    hpv_df = pd.DataFrame(data= np.array([[0]*len(HPV_col)]*len(df)) , columns=HPV_col)
    hpv_df.iloc[df['16'].isna()] = np.nan
    hpv_df['16'] = df['16']
    hpv_df['18'] = df['18']
    others = df.drop(columns=['16', '18']).sum(axis=1, skipna = False)
    hpv_df.loc[others.notna() & others.astype(bool), 'Others'] = 1
    pos = df.sum(axis=1, skipna = False)
    pos_flag = pos.astype(bool) & df['16'].notna()
    hpv_df.loc[pos.notna() & ~pos_flag, 'No'] = 1
    return hpv_df


def perpare_extra(main_data: pd.DataFrame, hpv_data: pd.DataFrame):
    hpv_data = prepare_hpv(hpv_data.copy())
    extra_df = pd.concat([main_data['Age'],main_data['Logul'],main_data['Pop'], hpv_data], axis = 1)
    extra_df["Age"] = pd.to_numeric(extra_df["Age"], errors="coerce").astype("Int64")
    return extra_df


def prepare_dataframe(main_data: pd.DataFrame, hpv_data: pd.DataFrame, imputation: Imputation|None= None):
    main_data = prepare_logul(main_data.copy())
    extra_data = perpare_extra(main_data, hpv_data.copy())

    if imputation is None:
        imputation = Imputation()
        extra_df = imputation.fit_transform(extra_data)
    else:
        extra_df = imputation.transform(extra_data)
    
    prepared_df = pd.concat([main_data[['Patient ID', 'jpg_file', 'Abnormality(Impression)', 'Abnormality(BX)']], extra_df], axis = 1)
    return prepared_df, imputation
    
