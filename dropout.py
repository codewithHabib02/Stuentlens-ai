import pandas as pd 
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt  
import seaborn as sns    
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance 
import os
os.environ['LOKY_MAX_CPU_COUNT']='4'
from sklearn.compose import ColumnTransformer  
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  
import numpy as np 
import time 
from sklearn.metrics import(
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   roc_auc_score
)

df=pd.read_csv('data.csv')

def print_section(title,data):
    print("\n" + '='*60)
    print(title)
    print("\n" + "=" * 60)
    print(data)
    
print_section("Head:", df.head())  
print_section("Info:",df.info)
print_section("Dtypes:", df.dtypes)
print_section("Cols:",df.columns.tolist())
print_section("Missing_values:",df.isnull().sum())
print_section("Missing_values:", df.isnull().sum().sum())
print_section("Missing_val_%:",df.isnull().sum()/len(df)*100)
print_section("Duplicated_values:",df.duplicated().sum())
print_section("Data_Description:",df.describe()) 

categorical_cols = [
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    'Gender',
    'Application mode',
    'Course'
]
df[categorical_cols]=df[categorical_cols].astype('category')

labelen=LabelEncoder()
df['Target']=labelen.fit_transform(df['Target'])


features = [
    # Academic performance — 1st semester
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',

    # Financial information
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',

    # Student background
    'Age at enrollment',
    'Admission grade',
    'Gender',

    # Enrollment information
    'Application mode',
    'Course'
]

x = df[features]
y = df['Target']

numeric_c=x.select_dtypes(include=['number']).columns.tolist()
categorical_c=x.select_dtypes(include=['object','category']).columns.tolist()

numeric_impute=Pipeline([
    ("impute",SimpleImputer(strategy='mean')),
    ("scaler",StandardScaler())
])

categorical_impute=Pipeline([
    ("impute",SimpleImputer(strategy='most_frequent')),
    ("encoder",OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(transformers=[
    ("num",numeric_impute,numeric_c),
    ("cat",categorical_impute,categorical_c)
])



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

collect=[]
models = {
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        
    
    ),

    "CatBoost": CatBoostClassifier(
         verbose=0,
         random_seed=42,
         auto_class_weights='Balanced'
        
    ),

    "SVC": SVC(probability=True, class_weight='balanced'),

    "Random Forest": RandomForestClassifier(
       n_estimators=200,
        random_state=42,
        class_weight='balanced'
      
    ),

    "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
          
    )
}

for name,model in models.items():
    pip=Pipeline([
        ("prepr",preprocessor),
        ("smote",SMOTE()),
        ("model",model)
    ])
    pip.fit(x_train,y_train)
    y_pred=pip.predict(x_test)
    y_proba=pip.predict_proba(x_test)
  
    collect.append({
        "model":   name,
        "Accuracy": accuracy_score(y_test,y_pred),
        "Precision": precision_score(y_test,y_pred,average='macro',zero_division=0), 
        "Recall":   recall_score(y_test,y_pred,average='macro',zero_division=0),
        "F1":      f1_score(y_test,y_pred,average='macro',zero_division=0),
        "Roc":   roc_auc_score(y_test,y_proba,multi_class='ovr',average='macro')
     
    })
    
fnal_df=pd.DataFrame(collect) 
print(fnal_df)



final_m=[]

final_model=Pipeline(steps=[
    ("prepr",preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model",SVC(probability=True))
])

param_grid = [
    {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["rbf"],
        "model__gamma": ["scale", "auto", 0.01, 0.1, 1]
    },
    {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear"]
    }
]

grid=GridSearchCV(
    final_model,
    param_grid,
    cv=2,
    scoring='accuracy',
    n_jobs=-1
    
)

grid.fit(x_train,y_train)

print(
    f"Best_params:   {grid.best_params_}\n"
    f"Best_score :   {grid.best_score_}\n"
    f"Best_estimator:{grid.best_estimator_}"
)

best_model=grid.best_estimator_
import joblib

joblib.dump(best_model, "student_outcome_model.pkl")

y_pred=best_model.predict(x_test)
y_proba=best_model.predict_proba(x_test)

final_m.append({
        "model":   'SVC',
        "Accuracy": accuracy_score(y_test,y_pred),
        "Precision": precision_score(y_test,y_pred,average='weighted',zero_division=0), 
        "Recall":   recall_score(y_test,y_pred,average='weighted',zero_division=0),
        "F1":      f1_score(y_test,y_pred,average='weighted',zero_division=0),
        "Roc":   roc_auc_score(y_test,y_proba,multi_class='ovr',average='weighted',labels=best_model.classes_)
     
    })
    
fnal_df=pd.DataFrame(final_m) 
print(fnal_df)

perm=permutation_importance(
    best_model,
    x_test,
    y_test,
    random_state=42,
    n_repeats=2,
    scoring='f1_macro',
    n_jobs=-1
)
imp=pd.DataFrame({
    "feature": x_test.columns,
    "importance": perm.importances_mean * 100
}).sort_values(by='importance',ascending=False)

print(imp)
