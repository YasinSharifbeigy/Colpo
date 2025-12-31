import pandas as pd
import numpy as np
from pandas import DataFrame
import os
from collections import Counter




def remove_duplicates(patients: list| str):
    if isinstance(patients, str):
        patients = os.listdir(patients)
    true_list = []
    duplicated_list = []
    for d in patients:
        if '(' in d:
            duplicated_list.append(d)
        else:
            true_list.append(d)
    return true_list, duplicated_list



extra_features_columns = ["Age", "Smoking", "Drink", "SD", "Married", "#Partner", "Pop"]
def extract_extra_features(ex_file: DataFrame):
    extra_df = pd.DataFrame(data= np.array([[np.nan]*len(extra_features_columns)]), columns=extra_features_columns)
    age_idx = 3
    
    if "Age" not in ex_file.iloc[age_idx,0]:
        i=0
        while "Age" not in ex_file.iloc[i,0]:
          i += 1
        age_idx = i
    smoke_idx = 22
    if "Smoking" not in ex_file.iloc[smoke_idx,0]:
        i=0
        while "Smoking" not in ex_file.iloc[i,0]:
          i += 1
        smoke_idx = i

    extra_df['Age'] = ex_file.iloc[age_idx+1,0]
    extra_df['Smoking'] = ex_file.iloc[smoke_idx+1,0]
    extra_df['Drink'] = ex_file.iloc[smoke_idx+1,1]
    extra_df['SD'] = ex_file.iloc[smoke_idx+1,2]
    extra_df['Married'] = ex_file.iloc[smoke_idx+1,3]
    extra_df['#Partner'] = ex_file.iloc[smoke_idx+1,4]

    return extra_df

procedure_columns = ["Date", "Adeno:conclusive", "AA:Color", "AA:Margin of aceto acid", "AA:Surface", "AA:Size", "Vessels:Punctuation",
                     "Vessels:Mosaics", "Vessels:cufed gland", "Vessels:Atypical", "Vessels:tree like vessels(on nabotian cyst)", "Logul",
                     "Erossion:lesion", "Erossion:clock", "Location of BX:BX1",
                     "Location of BX:BX2", "Diagnosis:Impression", "Diagnosis:BX1", "Diagnosis:BX2", "Diagnosis:ECC", "Diagnosis:POLYPS",
                     "Diagnosis:INFLMATION", "Diagnosis:ATROPHIC", "EXTRA INFO"]
def extract_procedure(ex_file: DataFrame):
    if ex_file.iloc[26,0] == "Procedure":
      pro_idx = 26
    else:
        i = 0
        while ex_file.iloc[i,0] != "Procedure":
          i += 1
        pro_idx = i
    i = 1
    while not pd.isna(ex_file.iloc[pro_idx+3+i,0]):
      i += 1
    # print(ex_file.iloc[pro_idx+3:pro_idx+3+i, :24].shape[1])
    data = ex_file.iloc[pro_idx+3:pro_idx+3+i, :24].values
    # print(data.shape, len(data))
    if(data.shape[-1]==23):
      data = np.concat((data, np.array([[np.nan]]*data.shape[0])), axis=1)
    # print(data.shape, data)
    procedure_df = pd.DataFrame(data= data, columns=procedure_columns)
    return procedure_df


def extract_HPV_types(ex_file):
    if ex_file.iloc[9,0] == "History":
        hist_idx = 9
    else:
        i = 0
        while ex_file.iloc[i,0] != "History":
          i += 1
        hist_idx = i
    i = hist_idx+2
    hpv_col = []

    # print(ex_file.iloc[i])
    while not pd.isna(ex_file.iloc[i,0]):
        # print((ex_file.iloc[i,0] == np.nan), type(ex_file.iloc[i,0]), "ffffffffff")
        hpv = ex_file.iloc[i,2]
        # print(hpv, type(hpv), hpv != np.nan)
        if not pd.isna(hpv):
            hpv = hpv.replace(" ", "")
            if ',' in hpv:
                hpv = hpv.split(",")
            elif '.' in hpv:
                hpv = hpv.split(".")
            elif hpv.isdigit():
                hpv = [hpv]
            else: hpv = [hpv]
        else:
            hpv = []
        hpv_col += hpv
        i += 1
    return hpv_col


def extract_all_hpv_types(data_path: str): #fata_path = os.path.join(root, "Cropped Folder")
    hpv_col = []
    hpv_dict = {}
    for folder in os.listdir():
        path = os.path.join(data_path, folder)
        if not os.path.isdir(path):
            continue
        records = os.listdir(path)
        ex_file = None
        for record in records:
            if record.endswith(".xlsx"):
                ex_name = record
                try:
                    ex_file = pd.read_excel(os.path.join(path, record), keep_default_na=False, na_values=[""], dtype= str)
                except:
                    ex_file = None
                    continue
                # print(path)

                hpv = extract_HPV_types(ex_file)
                hpv_col += hpv
        hpv_dict[folder] = hpv


hpv_col = ['3','6','11','14','16','18','26','31','32','33','35','36','39','40','41','42','43','44','45','51','52','53','54','55','56','58','59','61','62','66','67','68','70','72','73','74','75','80','81','82','83','84','89','90','91', 'others']
def extract_HPV_Pop(ex_file, date= None):
    HPV = pd.DataFrame(data= np.array([[0]*len(hpv_col)]), columns=hpv_col, dtype= float)
    if ex_file.iloc[9,0] == "History":
        hist_idx = 9
    else:
        i = 0
        while ex_file.iloc[i,0] != "History":
          i += 1
        hist_idx = i
    i = hist_idx+2
    # print(date, "gggggggggggggg")
    date_idx = -1
    while not pd.isna(ex_file.iloc[i,0]):
        if ex_file.iloc[i,0] == date:
            date_idx = i
            break
        i += 1

    if date_idx == -1:
        hpv = [np.nan]
        pop = np.nan
    else:
      hpv = ex_file.iloc[date_idx,2]
      pop = ex_file.iloc[date_idx,1]

    if not pd.isna(hpv):
        hpv = hpv.replace(" ", "")
        if ',' in hpv:
            hpv = hpv.split(",")
        elif '.' in hpv:
            hpv = hpv.split(".")
        elif hpv.isdigit():
            hpv = [hpv]
        elif 'oth' in hpv.lower() or 'pos' in hpv.lower():
            hpv = ['others']
        elif 'neg' in hpv.lower() or 'mal' in hpv.lower():
            hpv = []
        else:
            print(hpv)
    else:
        hpv = [np.nan]

    for h in hpv:
        if pd.isna(h):
            HPV.loc[:,:] = np.nan
        elif h.lower() == 'others':
            HPV.loc[0, 'others'] += 1
        elif h.isdigit() and not h in hpv_col:
            print(h , ' is not in the list')
            HPV.loc[0, 'others'] += 1
        else:
            HPV.loc[0,h] = 1
    return pop, HPV


def fill_normals(pro_data):
    flag = False
    if not pro_data['Diagnosis:Impression'].isna().item() and pro_data["Diagnosis:BX1"].isna().item():
        if pro_data['Diagnosis:Impression'][0].lower() == "normal": flag = True
    if not pro_data["Diagnosis:BX1"].isna().item():
        if pro_data["Diagnosis:BX2"].isna().item():
            if pro_data["Diagnosis:BX1"][0].lower() == "normal": flag = True
        else:
            if pro_data["Diagnosis:BX2"][0].lower() == "normal" and pro_data["Diagnosis:BX2"].item().lower() == "normal" : flag = True
    # print(flag)

    if flag:
        if pro_data['Adeno:conclusive'].isna().item():
            pro_data['Adeno:conclusive'] = "No"
        if pro_data['AA:Color'].isna().item():
            pro_data['AA:Color'] = "Pink"
        for cal in ["AA:Margin of aceto acid", "AA:Surface", "AA:Size", "Vessels:Punctuation", "Vessels:Mosaics", "Erossion:lesion"]:
            if pro_data[cal].isna().item():
                pro_data[cal] = "None"
        for cal in ["Vessels:cufed gland", "Vessels:Atypical", "Vessels:tree like vessels(on nabotian cyst)"]:
            if pro_data[cal].isna().item():
                pro_data[cal] = "No"
        if pro_data['Logul'].isna().item():
            pro_data['Logul'] = "Negative"
    return pro_data



def generate_raw_dataframe(patients: list|str, data_root:str= r"D:\Data\Cropped Folder"):
# from genericpath import isdir
    if isinstance(patients, str):
        data_root = patients
    true_data_list, _ = remove_duplicates(patients)        
    not_good = []
    data_list = []
    hpv_list = []
    data_columns = ["Patient ID", "jpg_file", "xlsx_file"] + procedure_columns + extra_features_columns + ['Abnormality(Impression)', 'Abnormality(BX)']
    bad_data_columns = ["Patient ID", "jpg_file", "xlsx_file", "Impression", "EXTRA INFO"]
    not_directory = []
    Imp_unique = {'Normal': 'Normal', 'Metaplasia': 'Metaplasia', 'Low grade ': 'Low grade', 'Erossion': 'Erosion', 'Intermadiate': 'Intermediate',
        'Intermediate': 'Intermediate', 'High grade ': 'High grade', 'Low grade': 'Low grade', 'High grade': 'High grade',
        'Eversion':'Eversion', 'Cancer ': 'Cancer', 'INTERMEDIAT': 'Intermediate', 'Errosion': 'Erosion', 'EROSION': 'Erosion',
        'NORMAL': 'Normal', 'LOW GRADE': 'Low grade'}
    for folder in true_data_list:
        path = os.path.join(data_root, folder)
        if not os.path.isdir(path):
            not_directory.append(folder)
            continue
        records = os.listdir(path)
        jpg_file = None
        ex_file = None
        # print(folder)
        rec_num = 0
        procedure_flag = False
        for record in records:
            jpg_file_null = None
            # print(path, record)
            if os.path.isdir(os.path.join(path, record)) and rec_num < 1:
                for img in os.listdir(os.path.join(path, record)):
                    if 'A1' in img:
                        jpg_file_null = os.path.join(record,img)
                    elif 'A' in img:
                        jpg_file_null = os.path.join(record,img)
            if jpg_file_null is not None:
                rec_num += 1
                jpg_file = jpg_file_null
            # print(folder, record, ex_file)
            if record.endswith(".xlsx") and procedure_flag == False:
                ex_name = record
                try:
                    ex_file = pd.read_excel(os.path.join(path, record), keep_default_na=False, na_values=[""], dtype= str)
                except:
                    ex_file = None
                    continue
                
                procedure_data = extract_procedure(ex_file)
                # print(procedure_data['Diagnosis:Impression'].isna(), "555")
                # print(procedure_data['EXTRA INFO'], "666")
                procedure_idx = 0
                for idx in range(procedure_data.shape[0]):
                    if procedure_data['Diagnosis:Impression'].isna()[idx] and procedure_data['Diagnosis:BX1'].isna()[idx] and procedure_data['Diagnosis:BX2'].isna()[idx]:
                        continue
                    if procedure_data['EXTRA INFO'][idx] == "DELET":
                        continue
                    procedure_idx = idx
                    procedure_flag = True
                    break
        if jpg_file is None or ex_file is None or "D" in folder:
            not_good.append([folder, jpg_file, ex_file is None, procedure_data['Diagnosis:Impression'][procedure_idx], procedure_data['EXTRA INFO'][procedure_idx]])
        elif not procedure_flag:
            not_good.append([folder, jpg_file, ex_file is None, procedure_data['Diagnosis:Impression'][procedure_idx], procedure_data['EXTRA INFO'][procedure_idx]])
        else:
            if not pd.isna(procedure_data['Diagnosis:Impression'][procedure_idx]):
                procedure_data.loc[procedure_idx, 'Diagnosis:Impression'] = Imp_unique[procedure_data['Diagnosis:Impression'][procedure_idx]]
            extra_feat = extract_extra_features(ex_file)
            pop, hpv = extract_HPV_Pop(ex_file, procedure_data['Date'][procedure_idx])
            extra_feat['Pop'] = pop
            data = [folder, jpg_file, ex_name] + procedure_data.loc[procedure_idx].tolist() + extra_feat.loc[0].tolist() + [np.nan, np.nan]
            data_list.append(data)
            hpv_list.append(hpv.loc[0].tolist())

    prepared_data = pd.DataFrame(data=data_list, columns=data_columns)
    hpv_data = pd.DataFrame(data=hpv_list, columns=hpv_col)
    not_good = pd.DataFrame(data=not_good, columns=bad_data_columns)

    return prepared_data, hpv_data, not_good

def add_labels(raw_df: DataFrame):
    flag_imp_normal = (raw_df["Diagnosis:Impression"] == "Normal") | (raw_df['Diagnosis:Impression'] == 'Eversion')
    flag_imp_nan = raw_df["Diagnosis:Impression"].isna()
    flag_bx_nan = raw_df["Diagnosis:BX1"].isna() & raw_df["Diagnosis:BX2"].isna()
    flag_bx_normal = (raw_df['Diagnosis:BX1'] == 'Normal') & (raw_df['Diagnosis:BX2'].isna() | (raw_df['Diagnosis:BX2'] == 'Normal'))

    raw_df['Abnormality(Impression)'] = (~flag_imp_normal & ~flag_imp_nan) | (flag_imp_nan & ~flag_bx_normal)
    raw_df['Abnormality(BX)'] = (~flag_bx_nan & ~flag_bx_normal) | (flag_bx_nan & ~flag_imp_normal)

    return raw_df

def category_table(raw_data: DataFrame, cols: list|str|None = None):
    if cols is None:
        cols = raw_data.columns
        cols = cols.drop(['Patient ID', 'jpg_file', 'xlsx_file', 'Date',  'EXTRA INFO', 'Age', 
                          'Erossion:clock', 'Location of BX:BX1', 'Location of BX:BX2', 'SD', 
                          '#Partner'])
    elif isinstance(cols, str): cols = [cols]
    
    rows = []

    
    for col in cols:
        counts = raw_data[col].value_counts(dropna=True)
        for k, v in counts.items():
            rows.append({
                "column": col,
                "category": "NaN" if pd.isna(k) else k,
                "count": int(v)
            })
        rows.append({
            "column": col,
            "category": "NaN", 
            "count": len(raw_data) - int(counts.sum())
        })

    return pd.DataFrame(rows)