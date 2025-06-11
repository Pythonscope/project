# well_log_model.py  –  complete, production-ready version
# ---------------------------------------------------------
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.impute        import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics       import mean_squared_error, r2_score

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
    # -------------------------------------------------- initialisation
    def __init__(self):
        self.data   = None
        self.feature_columns = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.extra_features  = ['RT_GR_Ratio', 'NPHI_RHOB_Crossplot']

        self.lithology_model  = None
        self.porosity_model   = None
        self.permeability_model = None
        self.saturation_model   = None

        self.scaler   = None
        self.metrics  = {}
        self.encoder  = LabelEncoder()

    # -------------------------------------------------- data handling
    def preprocess_data(self, path_or_buffer):
        """
        Read a CSV, impute numeric gaps and add engineered features.
        Accepts a file path, file-like object or Flask upload stream.
        """
        self.data = pd.read_csv(path_or_buffer)

        imputer = KNNImputer(n_neighbors=5)
        self.data[self.feature_columns] = imputer.fit_transform(
            self.data[self.feature_columns])

        # engineered ratios / cross-plots
        self.data['RT_GR_Ratio']         = self.data['RT']  / (self.data['GR']+1e-3)
        self.data['NPHI_RHOB_Crossplot'] = self.data['NPHI']* self.data['RHOB']
        return self.data

    def generate_targets(self):
        """Create synthetic targets for demo / debugging."""
        n = len(self.data)
        self.data['LITHOLOGY'] = np.random.choice(
            ['Sandstone', 'Limestone', 'Shale'], n)
        self.data['POROSITY']  = np.clip(
            0.25 - 0.001*self.data.GR + np.random.normal(0, .02, n), 0, .35)
        self.data['PERMEABILITY'] = np.clip(
            100*np.exp(-self.data.RT/50) + np.random.normal(0, 5, n), 0, 1000)
        self.data['WATER_SATURATION'] = np.clip(
            1 - self.data.POROSITY + np.random.normal(0, .05, n), 0, 1)

    # -------------------------------------------------- model training
    def train_models(self):
        if not HAS_SMOTE:
            raise ImportError("pip install imbalanced-learn  # required for SMOTE")

        X  = self.data[self.feature_columns + self.extra_features]
        yL = self.encoder.fit_transform(self.data['LITHOLOGY'])
        yΦ, yK, ySw = self.data.POROSITY, self.data.PERMEABILITY, self.data.WATER_SATURATION

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        X_bal, y_bal = SMOTE(random_state=42).fit_resample(Xs, yL)

        # --- classifier
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
            base, grid, cv=3, n_jobs=-1, scoring='accuracy'
        ).fit(X_bal, y_bal).best_estimator_

        cv_acc = cross_val_score(self.lithology_model, X_bal, y_bal,
                                 cv=5, scoring='accuracy').mean()

        # --- regressors
        def _reg(cls_rf_or_xgb):
            if HAS_XGB:
                return cls_rf_or_xgb(n_estimators=200, learning_rate=0.03)
            return cls_rf_or_xgb(n_estimators=200, random_state=42)

        self.porosity_model     = _reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, yΦ)
        self.permeability_model = _reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, yK)
        self.saturation_model   = _reg(XGBRegressor if HAS_XGB else RandomForestRegressor).fit(Xs, ySw)

        self.metrics = {'Lithology Accuracy (CV)': float(cv_acc)}

    # -------------------------------------------------- plotting
    def make_plot(self):
        """
        Return a five-track matplotlib Figure: GR, RT (log), NPHI, RHOB, lithology column.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        depth = self.data.DEPTH
        fig, axs = plt.subplots(1, 5, figsize=(14, 8), sharey=True)

        # GR
        axs[0].plot(self.data.GR, depth, 'g-', lw=.9)
        axs[0].set_xlim(0, 200);   axs[0].set_xlabel("GR (API)"); axs[0].set_title("GR")

        # RT (log)
        axs[1].semilogx(self.data.RT, depth, 'r-', lw=.9)
        axs[1].set_xlim(0.2, 2000); axs[1].set_xlabel("RT (Ω·m)"); axs[1].set_title("Resistivity")

        # NPHI
        axs[2].plot(self.data.NPHI, depth, 'b-', lw=.9)
        axs[2].set_xlim(0, .5);    axs[2].set_xlabel("NPHI");      axs[2].set_title("Neutron Φ")

        # RHOB
        axs[3].plot(self.data.RHOB, depth, 'k-', lw=.9)
        axs[3].set_xlim(1.95, 2.95); axs[3].set_xlabel("RHOB (g/cc)"); axs[3].set_title("Density")

        # Lithology block
        lith_cols = {'Sandstone':'gold', 'Limestone':'skyblue', 'Shale':'brown'}
        lith = self.data.get('LITHOLOGY',
                             pd.Series(['Unknown']*len(depth), index=self.data.index))
        for i in range(len(depth)-1):
            c = lith_cols.get(lith.iloc[i], 'grey')
            axs[4].fill_betweenx([depth.iloc[i], depth.iloc[i+1]], 0, 1, color=c, alpha=.7)
        axs[4].set_xlim(0,1); axs[4].set_xticks([]); axs[4].set_title("Lithology")

        for ax in axs:
            ax.invert_yaxis(); ax.grid(True, alpha=.3); ax.set_ylabel("Depth (ft)")

        fig.tight_layout()
        return fig

    # -------------------------------------------------- recommendations
    def generate_recommendations(self):
        if self.scaler is None:
            raise RuntimeError("Run train_models() first")

        Xs   = self.scaler.transform(self.data[self.feature_columns + self.extra_features])
        lith = self.encoder.inverse_transform(self.lithology_model.predict(Xs))
        Φ    = self.porosity_model.predict(Xs)
        k    = self.permeability_model.predict(Xs)
        Sw   = self.saturation_model.predict(Xs)

        df = pd.DataFrame({'DEPTH': self.data.DEPTH,
                           'LITH' : lith,
                           'PHI'  : Φ,
                           'PERM' : k,
                           'SW'   : Sw})

        pay   = (df.PHI>.20) & (df.PERM>100) & (df.SW<.60)
        water = df.SW>.70
        frac  = df.PERM.between(50,100)
        sand  = df.LITH == 'Sandstone'
        rng   = lambda m: f"{df[m].DEPTH.min():.1f}-{df[m].DEPTH.max():.1f} ft" if m.any() else "-"

        msg  = "=== AI-Powered Recommendations ===\n"
        msg += f"Avg Φ {Φ.mean():.2%} | Avg k {k.mean():.1f} mD | Avg Sw {Sw.mean():.2%}\n\n"
        msg += f"High-quality pay   : {pay.sum():4d}  @ {rng(pay)}\n"
        msg += f"High water risk    : {water.sum():4d}  @ {rng(water)}\n"
        msg += f"Frac candidates    : {frac.sum():4d}  @ {rng(frac)}\n"
        msg += f"Sandstone sections : {sand.sum():4d}  @ {rng(sand)}\n"
        return msg
