# ================================================================
# well_log_model.py  ─  PRO edition
# ================================================================
import time, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.impute            import KNNImputer
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble          import GradientBoostingRegressor
from sklearn.model_selection   import GridSearchCV, train_test_split
from sklearn.metrics           import accuracy_score, r2_score

try:                        # optional – only used if installed
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    warnings.warn("imblearn not installed – SMOTE disabled")

# ----------------------------------------------------------------
class ProWellLogInterpreter:
    """Professional-grade, resource-aware well-log AI interpreter"""
    # ───────────────────────────────────────────────────────────
    def __init__(self):
        self.data    : pd.DataFrame | None = None
        self.features: list[str] = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.models  = {}
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self.metrics = {}

    # ───────────────────────────────────────────────────────────
    # 1. DATA INGEST / PRE-PROCESS
    def preprocess_data(self, csv: str | bytes | bytearray):
        """Load CSV, clean, impute, engineer features; return dataframe."""
        t0 = time.time()
        df = pd.read_csv(csv)

        # basic sanity
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # clip obvious outliers to 1st-99th percentiles
        for col in self.features:
            lo, hi = np.percentile(df[col].dropna(), [1, 99])
            df[col] = df[col].clip(lo, hi)

        # KNN impute (handles non-linear gaps better than mean/median)
        df[self.features] = KNNImputer(n_neighbors=5).fit_transform(df[self.features])

        # ── feature engineering ──
        df['RT_GR_Ratio']          = df['RT']   / (df['GR']   + 1e-3)
        df['NPHI_RHOB_Crossplot']  = df['NPHI'] * df['RHOB']
        df['GR_log']               = np.log1p(df['GR'])
        df['RT_log']               = np.log1p(df['RT'])
        df['Deep_Resistivity']     = np.log1p(df['RT'] * df['PEF'])  # toy example

        self.extra = ['RT_GR_Ratio','NPHI_RHOB_Crossplot','GR_log','RT_log',
                      'Deep_Resistivity']
        self.all_feat = self.features + self.extra
        self.data = df

        print(f"PREPROCESS: {len(df)} rows, {len(self.all_feat)} features "
              f"({time.time()-t0:.2f}s)")
        return df

    # ───────────────────────────────────────────────────────────
    # 2. SYNTHETIC TARGET GENERATION (for demo environments)
    def generate_targets(self, seed: int = 42):
        """Create demo lithology & petrophysical targets."""
        if self.data is None:
            raise RuntimeError("Call preprocess_data() first")

        np.random.seed(seed)
        n = len(self.data)

        # lithology rule-based mock-up
        def lithologic(row):
            if row.GR > 75 and row.RT < 10:                     return 'Shale'
            if row.GR < 30 and row.RT > 100 and row.NPHI < .15: return 'Sandstone'
            if row.GR < 50 and row.RT > 20:                     return 'Sandstone'
            if row.RHOB > 2.7 and row.PEF > 4:                  return 'Limestone'
            return 'Shale'

        self.data['LITHOLOGY'] = self.data.apply(lithologic, axis=1)

        # reservoir properties
        phi  = 0.40 - (self.data.RHOB - 1.5) * 0.18 + np.random.normal(0, .02, n)
        k    = np.exp( 8*phi - 2) * (1 / (self.data.RT+1)) * np.random.lognormal(0,.4,n)
        sw   = 0.25 + 0.55/(1+np.exp((self.data.RT-60)/18)) + np.random.normal(0,.04,n)

        self.data['POROSITY']        = np.clip(phi , .05, .35)
        self.data['PERMEABILITY']    = np.clip(k   , .1 , 1500)
        self.data['WATER_SATURATION']= np.clip(sw  , .15, 1.0)

    # ───────────────────────────────────────────────────────────
    # 3. TRAIN
    def train_models(self):
        if self.data is None:
            raise RuntimeError("Data not loaded")

        X = self.data[self.all_feat].values
        Xs = self.scaler.fit_transform(X)

        # --- lithology classifier ------------------------------------------------
        y_cls = self.encoder.fit_transform(self.data['LITHOLOGY'])
        X_train, X_val, y_train, y_val = train_test_split(
            Xs, y_cls, stratify=y_cls, test_size=0.2, random_state=42)

        # optional SMOTE to balance classes
        if HAS_SMOTE:
            try:
                X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
            except Exception as e:
                warnings.warn(f"SMOTE skipped: {e}")

        grid = GridSearchCV(
            RandomForestClassifier(),
            param_grid={'n_estimators':[100,200],
                        'max_depth'   :[6,10]},
            cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        acc = accuracy_score(y_val, clf.predict(X_val))

        # --- regression models ---------------------------------------------------
        def reg(model, y):      # helper to fit & score
            r = model.fit(X_train, y_train_reg := y[train_idx])
            r2 = r2_score(y_val_reg := y[val_idx], r.predict(X_val))
            return r, r2

        # index masks for fast scoring
        train_idx, val_idx = X_train[:,0].shape[0], X_val[:,0].shape[0]
        full_idx = np.arange(len(self.data))
        train_idx = np.isin(full_idx, train_idx)
        val_idx   = ~train_idx

        y_phi  = self.data['POROSITY'        ].values
        y_perm = self.data['PERMEABILITY'    ].values
        y_sw   = self.data['WATER_SATURATION'].values

        rf_reg = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        phi_model , r2_phi  = reg(RandomForestRegressor(**rf_reg), y_phi )
        perm_model, r2_perm = reg(GradientBoostingRegressor(random_state=42), y_perm)
        sw_model  , r2_sw   = reg(RandomForestRegressor(**rf_reg), y_sw  )

        # store
        self.models  = dict(lithology=clf, porosity=phi_model,
                            permeability=perm_model, saturation=sw_model)
        self.metrics = dict(lithology_acc=f"{acc:.3f}",
                            r2_phi=f"{r2_phi:.3f}",
                            r2_perm=f"{r2_perm:.3f}",
                            r2_sw=f"{r2_sw:.3f}")
        print("TRAINING summary:", self.metrics)

    # ───────────────────────────────────────────────────────────
    # 4. PLOT  (headline removed)
    def make_plot(self):
        if self.data is None or 'DEPTH' not in self.data.columns:
            raise RuntimeError("DEPTH column missing")

        depth = self.data['DEPTH']
        fig, ax = plt.subplots(1, 5, figsize=(15,8), sharey=True)

        ax[0].plot(self.data.GR , depth, 'g-'); ax[0].set_xlim(0,200); ax[0].set_title('GR')
        ax[1].semilogx(self.data.RT, depth,'r-'); ax[1].set_xlim(.2,2000); ax[1].set_title('RT')
        ax[2].plot(self.data.NPHI, depth,'b-'); ax[2].set_xlim(0,.5); ax[2].set_title('NPHI')
        ax[3].plot(self.data.RHOB, depth,'k-'); ax[3].set_xlim(1.95,2.95); ax[3].set_title('RHOB')

        colors = {'Sandstone':'#FFD700','Limestone':'#87CEEB','Shale':'#A0522D'}
        for i in range(len(depth)-1):
            col = colors.get(self.data.LITHOLOGY.iloc[i],'grey')
            ax[4].fill_betweenx(depth.iloc[i:i+2],0,1,color=col,alpha=.8)
        ax[4].set_title('Lithology'); ax[4].set_xticks([]); ax[4].set_xlim(0,1)
        from matplotlib.patches import Patch
        ax[4].legend([Patch(fc=c) for c in colors.values()],
                     colors.keys(), loc='center left', bbox_to_anchor=(1,0.5))
        for a in ax: a.invert_yaxis(); a.grid(alpha=.3); a.set_ylabel('Depth (ft)')
        plt.tight_layout()
        return fig

    # ───────────────────────────────────────────────────────────
    # 5. TEXT REPORT  (numbered)
    def generate_recommendations(self):
        if not self.models:
            raise RuntimeError("Train models first")

        Xs = self.scaler.transform(self.data[self.all_feat])
        lith = self.encoder.inverse_transform(self.models['lithology'].predict(Xs))
        phi  = self.models['porosity'].predict(Xs)
        perm = self.models['permeability'].predict(Xs)
        sw   = self.models['saturation'].predict(Xs)

        df = pd.DataFrame(dict(DEPTH=self.data.DEPTH, LITH=lith,
                               PHI=phi, PERM=perm, SW=sw))
        pay = (df.PHI>.15)&(df.PERM>50)&(df.SW<.6)

        txt  = "════════════════════════ RESERVOIR ANALYSIS ════════════════════════\n"
        txt += f"1. Interval analysed : {df.DEPTH.min():.0f}-{df.DEPTH.max():.0f} ft\n"
        txt += f"2. Main lithology    : {df.LITH.mode()[0]}\n"
        txt += f"3. Net pay           : {pay.sum()} ft ({pay.mean():.1%})\n"
        txt += f"4. Avg Porosity      : {phi.mean():.1%}\n"
        txt += f"5. Avg Perm          : {perm.mean():.1f} mD\n"
        txt += f"6. Avg Water Sat     : {sw.mean():.1%}\n\n"

        txt += "RECOMMENDATIONS\n---------------\n"
        if pay.sum()>20:
            txt += "1. Excellent reservoir quality – pursue primary completion.\n"
        elif pay.sum()>10:
            txt += "1. Moderate pay – selective perforation recommended.\n"
        else:
            txt += "1. Marginal pay – consider advanced stimulation or sidetrack.\n"
        txt += "2. Perform pressure/buildup test to confirm drive mechanism.\n"
        txt += "3. Run cased-hole saturation monitoring logs during production.\n"
        txt += "═════════════════════════════════════════════════════════════════════"
        return txt
# ----------------------------------------------------------------
# factory for fast import
def get_interpreter():
    return ProWellLogInterpreter()
