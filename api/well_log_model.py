# ================================================================
# well_log_model.py — “PRO” edition
# Professional yet resource-aware AI interpreter for well-log data
# ================================================================
import time, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.impute            import KNNImputer
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble          import GradientBoostingRegressor
from sklearn.model_selection   import GridSearchCV, train_test_split
from sklearn.metrics           import accuracy_score, r2_score

try:                                # optional – class-imbalance helper
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    warnings.warn("imblearn not installed  ➜  SMOTE disabled")


# ────────────────────────────────────────────────────────────────────
class ProWellLogInterpreter:
    """Professional-grade, deploy-friendly well-log AI module"""
    # --------------------------------------------------------------
    def __init__(self):
        self.data:    pd.DataFrame | None = None
        self.features                    = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.extra:   list[str]          = []
        self.all_feat: list[str]         = []
        self.models                       = {}
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self.metrics = {}

    # --------------------------------------------------------------
    # 1)  LOAD  +  PREPROCESS
    def preprocess_data(self, csv: str | bytes | bytearray):
        """Read CSV, clean, impute, engineer features, return DataFrame."""
        t0 = time.time()
        df = pd.read_csv(csv)

        # sanity check
        miss = [c for c in self.features if c not in df.columns]
        if miss:
            raise ValueError(f"Missing required columns {miss}")

        # clip extreme outliers to 1–99 percentiles
        for col in self.features:
            lo, hi = np.percentile(df[col].dropna(), [1, 99])
            df[col] = df[col].clip(lo, hi)

        # KNN imputation (handles non-linear gaps)
        df[self.features] = KNNImputer(n_neighbors=5).fit_transform(df[self.features])

        # feature engineering
        df['RT_GR_Ratio']         = df['RT']   / (df['GR'] + 1e-3)
        df['NPHI_RHOB_Cross']     = df['NPHI'] * df['RHOB']
        df['GR_log']              = np.log1p(df['GR'])
        df['RT_log']              = np.log1p(df['RT'])
        df['Deep_Resistivity']    = np.log1p(df['RT'] * df['PEF'])

        self.extra    = ['RT_GR_Ratio','NPHI_RHOB_Cross',
                         'GR_log','RT_log','Deep_Resistivity']
        self.all_feat = self.features + self.extra
        self.data     = df

        print(f"PREPROCESS ✔  rows={len(df)}  cols={len(self.all_feat)}  "
              f"time={time.time()-t0:.2f}s")
        return df

    # --------------------------------------------------------------
    # 2)  SYNTHETIC TARGETS (quick demo generator)
    def generate_targets(self, seed: int = 42):
        if self.data is None:
            raise RuntimeError("Call preprocess_data first")

        np.random.seed(seed)
        n = len(self.data)

        # simple lithology rules
        def lith(row):
            if row.GR>75 and row.RT<10:                      return 'Shale'
            if row.GR<30 and row.RT>100 and row.NPHI<.15:    return 'Sandstone'
            if row.GR<50 and row.RT>20:                      return 'Sandstone'
            if row.RHOB>2.7 and row.PEF>4:                   return 'Limestone'
            return 'Shale'

        self.data['LITHOLOGY'] = self.data.apply(lith, axis=1)

        phi  = 0.40 - (self.data.RHOB-1.5)*0.18 + np.random.normal(0,.02,n)
        k    = np.exp(8*phi-2)*(1/(self.data.RT+1))*np.random.lognormal(0,.4,n)
        sw   = 0.25 + 0.55/(1+np.exp((self.data.RT-60)/18)) + np.random.normal(0,.04,n)

        self.data['POROSITY']         = np.clip(phi ,.05,.35)
        self.data['PERMEABILITY']     = np.clip(k   ,.1 ,1500)
        self.data['WATER_SATURATION'] = np.clip(sw  ,.15,1.0)

    # --------------------------------------------------------------
    # 3)  TRAIN MODELS
    def train_models(self):
        if self.data is None:
            raise RuntimeError("Data missing")
        X   = self.data[self.all_feat].values
        Xs  = self.scaler.fit_transform(X)
        y_c = self.encoder.fit_transform(self.data['LITHOLOGY'])

        Xtr,Xva,ytr,yva = train_test_split(Xs,y_c,test_size=.2,
                                           stratify=y_c, random_state=42)

        # SMOTE (optional)
        if HAS_SMOTE:
            try:
                Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr,ytr)
            except Exception as e:
                warnings.warn(f"SMOTE skipped: {e}")

        # grid-searched random-forest classifier
        grid = GridSearchCV(RandomForestClassifier(),
                            {'n_estimators':[100,200],'max_depth':[6,10]},
                            cv=3,n_jobs=-1,scoring='accuracy')
        grid.fit(Xtr,ytr)
        clf  = grid.best_estimator_
        acc  = accuracy_score(yva, clf.predict(Xva))

        # regression models
        rf_reg = dict(n_estimators=200,max_depth=8,random_state=42,n_jobs=-1)
        phi_m  = RandomForestRegressor(**rf_reg).fit(Xtr, self.data.POROSITY.values[ytr.index])
        perm_m = GradientBoostingRegressor(random_state=42).fit(Xtr,self.data.PERMEABILITY.values[ytr.index])
        sw_m   = RandomForestRegressor(**rf_reg).fit(Xtr, self.data.WATER_SATURATION.values[ytr.index])

        self.models  = dict(lithology=clf, porosity=phi_m,
                            permeability=perm_m, saturation=sw_m)
        self.metrics = dict(lithology_acc=f"{acc:.3f}")
        print("TRAIN ✔  accuracy=", self.metrics['lithology_acc'])

    # --------------------------------------------------------------
    # 4)  PLOT  (no headline title)
    def make_plot(self):
        if self.data is None or 'DEPTH' not in self.data:
            raise RuntimeError("DEPTH column missing")
        depth=self.data.DEPTH
        fig,ax=plt.subplots(1,5,figsize=(15,8),sharey=True)
        ax[0].plot(self.data.GR, depth,'g-'); ax[0].set_xlim(0,200);   ax[0].set_title('GR')
        ax[1].semilogx(self.data.RT,depth,'r-');ax[1].set_xlim(.2,2000);ax[1].set_title('RT')
        ax[2].plot(self.data.NPHI,depth,'b-'); ax[2].set_xlim(0,.5);   ax[2].set_title('NPHI')
        ax[3].plot(self.data.RHOB,depth,'k-'); ax[3].set_xlim(1.95,2.95);ax[3].set_title('RHOB')

        colors={'Sandstone':'#FFD700','Limestone':'#87CEEB','Shale':'#A0522D'}
        for i in range(len(depth)-1):
            col=colors.get(self.data.LITHOLOGY.iat[i],'grey')
            ax[4].fill_betweenx(depth.iloc[i:i+2],0,1,color=col,alpha=.8)
        ax[4].set_title('Lithology'); ax[4].set_xlim(0,1); ax[4].set_xticks([])
        from matplotlib.patches import Patch
        legend=[Patch(fc=c,label=l) for l,c in colors.items()
                if l in self.data.LITHOLOGY.unique()]
        ax[4].legend(handles=legend,loc='center left',bbox_to_anchor=(1,0.5))
        for a in ax: a.invert_yaxis(); a.grid(alpha=.3); a.set_ylabel('Depth (ft)')
        plt.tight_layout(); return fig

    # --------------------------------------------------------------
    # 5)  TEXT REPORT (numbered, ready for download)
    def generate_recommendations(self):
        if not self.models: raise RuntimeError("Train first")
        Xs = self.scaler.transform(self.data[self.all_feat])
        lith = self.encoder.inverse_transform(self.models['lithology'].predict(Xs))
        phi  = self.models['porosity'    ].predict(Xs)
        perm = self.models['permeability'].predict(Xs)
        sw   = self.models['saturation'  ].predict(Xs)

        df=pd.DataFrame(dict(DEPTH=self.data.DEPTH,LITH=lith,PHI=phi,PERM=perm,SW=sw))
        pay=(df.PHI>.15)&(df.PERM>50)&(df.SW<.6)

        txt  = "════════════════════════ RESERVOIR ANALYSIS ════════════════════════\n"
        txt += f"1. Interval analysed : {df.DEPTH.min():.0f}-{df.DEPTH.max():.0f} ft\n"
        txt += f"2. Dominant lithology: {df.LITH.mode()[0]}\n"
        txt += f"3. Net-to-gross       : {pay.mean():.1%}  ({pay.sum()} ft pay)\n"
        txt += f"4. Avg. Porosity      : {phi.mean():.1%}\n"
        txt += f"5. Avg. Permeability  : {perm.mean():.1f} mD\n"
        txt += f"6. Avg. Water Sat     : {sw.mean():.1%}\n\n"
        txt += "RECOMMENDATIONS\n---------------\n"
        txt += ("1. Excellent pay – proceed with primary completion.\n" if pay.sum()>20 else
                "1. Moderate pay  – selective perforations advised.\n" if pay.sum()>10 else
                "1. Marginal pay  – further petrophysical review needed.\n")
        txt += "2. Perform pressure-buildup test to confirm drive mechanism.\n"
        txt += "3. Acquire cased-hole saturation monitoring during production.\n"
        txt += "═════════════════════════════════════════════════════════════════════"
        return txt


# factory helper for external import
def get_interpreter() -> ProWellLogInterpreter:
    return ProWellLogInterpreter()
