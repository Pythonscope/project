import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ────────────────────────────────────────────────────────────────
class WellLogInterpreter:
    def __init__(self):
        self.data               = None
        self.feature_columns    = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.lithology_model    = None
        self.porosity_model     = None
        self.permeability_model = None
        self.saturation_model   = None
        self.scaler             = None
        self.encoder            = LabelEncoder()
        self.metrics            = {}

    # ────────────────────────────────────────────────────────────
    # 1. LOAD  + PRE-PROCESS
    def preprocess_data(self, path_or_buffer):
        self.data = pd.read_csv(path_or_buffer)

        missing = [c for c in self.feature_columns if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # mean imputation (fast)
        imputer = SimpleImputer(strategy='mean')
        self.data[self.feature_columns] = imputer.fit_transform(
            self.data[self.feature_columns])

        # simple engineered features
        self.data['RT_GR_Ratio']         = self.data['RT'] / (self.data['GR'] + 1e-3)
        self.data['NPHI_RHOB_Crossplot'] = self.data['NPHI'] * self.data['RHOB']
        return self.data

    # ────────────────────────────────────────────────────────────
    # 2. SYNTHETIC TARGETS
    def generate_targets(self):
        if self.data is None:
            raise ValueError("No data loaded – upload first")

        n = len(self.data)
        np.random.seed(42)
        self.data['LITHOLOGY']       = np.random.choice(['Sandstone',
                                                         'Limestone',
                                                         'Shale'], n)
        self.data['POROSITY']        = np.clip(
            0.25 - 0.001*self.data['GR'] + np.random.normal(0, 0.02, n), 0, 0.35)
        self.data['PERMEABILITY']    = np.clip(
            100*np.exp(-self.data['RT']/50)+np.random.normal(0, 5, n), 0, 1000)
        self.data['WATER_SATURATION'] = np.clip(
            1 - self.data['POROSITY'] + np.random.normal(0, 0.05, n), 0, 1)

    # ────────────────────────────────────────────────────────────
    # 3. LIGHT-WEIGHT TRAINING
    def train_models(self):
        if self.data is None:
            raise ValueError("No data loaded")

        targets = ['LITHOLOGY', 'POROSITY', 'PERMEABILITY', 'WATER_SATURATION']
        if any(t not in self.data.columns for t in targets):
            raise ValueError("Target columns missing – run generate_targets()")

        X   = self.data[self.feature_columns].copy()
        y_L = self.encoder.fit_transform(self.data['LITHOLOGY'])
        y_P = self.data['POROSITY'].values
        y_K = self.data['PERMEABILITY'].values
        y_S = self.data['WATER_SATURATION'].values

        # scale once
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        rf_cls = RandomForestClassifier(n_estimators=20, max_depth=4,
                                        random_state=42, n_jobs=1)
        rf_reg = dict(n_estimators=20, max_depth=4, random_state=42, n_jobs=1)

        self.lithology_model    = rf_cls.fit(Xs, y_L)
        self.porosity_model     = RandomForestRegressor(**rf_reg).fit(Xs, y_P)
        self.permeability_model = RandomForestRegressor(**rf_reg).fit(Xs, y_K)
        self.saturation_model   = RandomForestRegressor(**rf_reg).fit(Xs, y_S)

        self.metrics = {"Samples": len(self.data),
                        "Model":   "Ultra-light RandomForest"}

    # ────────────────────────────────────────────────────────────
    # 4. PROFESSIONAL PLOT  (title removed)
    def make_plot(self):
        if self.data is None or 'DEPTH' not in self.data.columns:
            raise ValueError("DEPTH column missing")

        depth = self.data['DEPTH']
        fig, ax = plt.subplots(1, 5, figsize=(14, 8), sharey=True)

        # track-1 GR
        ax[0].plot(self.data['GR'], depth, 'g-')
        ax[0].set_xlim(0, 200); ax[0].set_xlabel('GR (API)')
        ax[0].set_title('GR'); ax[0].grid(alpha=.3)

        # track-2 RT
        ax[1].semilogx(self.data['RT'], depth, 'r-')
        ax[1].set_xlim(0.2, 2000); ax[1].set_xlabel('RT (Ω·m)')
        ax[1].set_title('RT'); ax[1].grid(alpha=.3)

        # track-3 NPHI
        ax[2].plot(self.data['NPHI'], depth, 'b-')
        ax[2].set_xlim(0, 0.5); ax[2].set_xlabel('NPHI')
        ax[2].set_title('NPHI'); ax[2].grid(alpha=.3)

        # track-4 RHOB
        ax[3].plot(self.data['RHOB'], depth, 'k-')
        ax[3].set_xlim(1.95, 2.95); ax[3].set_xlabel('RHOB (g/cc)')
        ax[3].set_title('RHOB'); ax[3].grid(alpha=.3)

        # track-5 Lithology
        colors = {'Sandstone':'#FFD700','Limestone':'#87CEEB','Shale':'#A0522D'}
        for i in range(len(depth)-1):
            lith = self.data['LITHOLOGY'].iat[i]
            ax[4].fill_betweenx(depth.iloc[i:i+2], 0, 1,
                                color=colors.get(lith,'grey'), alpha=.8)
        ax[4].set_xlim(0,1); ax[4].set_xticks([]); ax[4].set_title('Lithology')

        # legend
        from matplotlib.patches import Patch
        patches=[Patch(fc=c, label=l) for l,c in colors.items()]
        ax[4].legend(handles=patches, loc='center left', bbox_to_anchor=(1.04,0.5))

        # formatting
        for a in ax:
            a.invert_yaxis(); a.set_ylabel('Depth (ft)')
        plt.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────
    # 5. NUMBERED RECOMMENDATION REPORT  (returned as txt)
    def generate_recommendations(self):
        if self.scaler is None:
            raise RuntimeError("Run train_models() first")

        Xs = self.scaler.transform(self.data[self.feature_columns])
        lith = self.encoder.inverse_transform(
            self.lithology_model.predict(Xs))
        phi  = self.porosity_model.predict(Xs)
        perm = self.permeability_model.predict(Xs)
        sw   = self.saturation_model.predict(Xs)

        df = pd.DataFrame({'DEPTH':self.data['DEPTH'],
                           'LITH':lith,'PHI':phi,'PERM':perm,'SW':sw})
        pay = (df.PHI>0.12)&(df.PERM>10)&(df.SW<0.65)

        # ---------- plain-text report ----------
        txt  =  "════════════════════════ RESERVOIR ANALYSIS ════════════════════════\n"
        txt += f"1. Interval analysed : {df.DEPTH.min():.0f}-{df.DEPTH.max():.0f} ft\n"
        txt += f"2. Net-to-gross ratio: {pay.mean():.1%}\n"
        txt += f"3. Avg. Porosity     : {phi.mean():.1%}\n"
        txt += f"4. Avg. Permeability : {perm.mean():.1f} mD\n"
        txt += f"5. Avg. Water Sat.   : {sw.mean():.1%}\n"
        txt += "\nRECOMMENDATIONS\n---------------\n"
        if pay.sum()>20:
            txt += "1. Good pay detected – perforate main sand units.\n"
        elif pay.sum()>10:
            txt += "1. Moderate pay – consider selective completion.\n"
        else:
            txt += "1. Marginal pay – further analysis advised.\n"
        txt += "2. Acquire pressure data to confirm drive mechanism.\n"
        txt += "3. Run production test before final completion design.\n"
        txt += "═════════════════════════════════════════════════════════════════════"
        return txt
