import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


class WellLogInterpreter:
    def __init__(self):
        self.data = None
        self.feature_columns = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.extra_features  = ['RT_GR_Ratio', 'NPHI_RHOB_Crossplot']
        self.lithology_model = self.porosity_model = \
        self.permeability_model = self.saturation_model = None
        self.scaler = None
        self.lithology_encoder = LabelEncoder()
        self.metrics = {}

    # ------------------------------------------------------------------  data
    def preprocess_data(self, path):
        self.data = pd.read_csv(path)
        # FIX 1: Correct KNNImputer usage
        imputer = KNNImputer(n_neighbors=5)  # <- Fixed: n_neighbors parameter
        self.data[self.feature_columns] = imputer.fit_transform(
            self.data[self.feature_columns])
        # FIX 2: Add feature engineering here (before using extra_features)
        self.data['RT_GR_Ratio']        = self.data['RT'] / (self.data['GR'] + 1e-3)
        self.data['NPHI_RHOB_Crossplot']= self.data['NPHI'] * self.data['RHOB']
        return self.data

    def generate_targets(self):
        n = len(self.data)
        self.data['LITHOLOGY']       = np.random.choice(
            ['Sandstone', 'Limestone', 'Shale'], n)
        self.data['POROSITY']        = np.clip(0.25 - 0.001*self.data['GR'] +
                                               np.random.normal(0,0.02,n), 0, .35)
        self.data['PERMEABILITY']    = np.clip(100*np.exp(-self.data['RT']/50) +
                                               np.random.normal(0,5,n), 0, 1000)
        self.data['WATER_SATURATION']= np.clip(1-self.data['POROSITY'] +
                                               np.random.normal(0,.05,n), 0, 1)

    # ------------------------------------------------------------------  train
    def train_models(self):
        if not HAS_SMOTE:
            raise ImportError('pip install imbalanced-learn')

        X = self.data[self.feature_columns + self.extra_features]
        y_lith = self.lithology_encoder.fit_transform(self.data['LITHOLOGY'])
        y_por, y_perm, y_sw = (self.data['POROSITY'],
                               self.data['PERMEABILITY'],
                               self.data['WATER_SATURATION'])

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        X_bal, y_bal = SMOTE(random_state=42).fit_resample(Xs, y_lith)

        if HAS_XGB:
            base = XGBClassifier(objective='multi:softprob',
                                 use_label_encoder=False,
                                 eval_metric='mlogloss')
            grid = {'n_estimators':[100,200], 'max_depth':[3,5],
                    'learning_rate':[0.03,0.1]}
        else:
            base = RandomForestClassifier(random_state=42)
            grid = {'n_estimators':[100,200], 'max_depth':[5,10]}

        self.lithology_model = GridSearchCV(
            base, grid, cv=3, scoring='accuracy', n_jobs=-1
        ).fit(X_bal, y_bal).best_estimator_

        cv_acc = cross_val_score(self.lithology_model, X_bal, y_bal,
                                 cv=5, scoring='accuracy').mean()

        def reg(cls):   # helper to choose RF or XGB regressor
            if HAS_XGB:
                return cls(n_estimators=200, learning_rate=0.03)
            return cls(n_estimators=200, random_state=42)

        self.porosity_model     = reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, y_por)
        self.permeability_model = reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, y_perm)
        self.saturation_model   = reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, y_sw)

        self.metrics = {'Lithology Accuracy (CV)': cv_acc}

    # ------------------------------------------------------------------  recommend
    def generate_recommendations(self):
        Xs = self.scaler.transform(self.data[self.feature_columns + self.extra_features])

        lith = self.lithology_encoder.inverse_transform(
            self.lithology_model.predict(Xs))
        por  = self.porosity_model.predict(Xs)
        perm = self.permeability_model.predict(Xs)
        sw   = self.saturation_model.predict(Xs)

        df = pd.DataFrame({'DEPTH':self.data['DEPTH'],
                           'LITH':lith,'PHI':por,'PERM':perm,'SW':sw})

        pay   = (df['PHI']>.20)&(df['PERM']>100)&(df['SW']<.6)
        water = df['SW']>.70
        frac  = df['PERM'].between(50,100)
        sand  = df['LITH']=='Sandstone'
        rng   = lambda m: f"{df[m]['DEPTH'].min():.1f}-{df[m]['DEPTH'].max():.1f} ft" if m.any() else "-"

        msg  = "=== AI-Powered Recommendations ===\n"
        msg += f"Avg Φ {por.mean():.2%} | Avg k {perm.mean():.1f} mD | Avg Sw {sw.mean():.2%}\n\n"
        msg += f"High-quality pay   : {pay.sum():3d}  @ {rng(pay)}\n"
        msg += f"High water risk    : {water.sum():3d}  @ {rng(water)}\n"
        msg += f"Frac candidates    : {frac.sum():3d}  @ {rng(frac)}\n"
        msg += f"Sandstone sections : {sand.sum():3d}  @ {rng(sand)}\n"
        return msg
