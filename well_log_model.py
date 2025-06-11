import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
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
        self.extra_features = ['RT_GR_Ratio', 'NPHI_RHOB_Crossplot']
        self.lithology_model = None
        self.porosity_model = None
        self.permeability_model = None
        self.saturation_model = None
        self.scaler = None
        self.encoder = LabelEncoder()
        self.metrics = {}

    def preprocess_data(self, path_or_buffer):
        """Load CSV, impute missing values, add engineered features."""
        self.data = pd.read_csv(path_or_buffer)
        
        # Check if required columns exist
        missing_cols = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        self.data[self.feature_columns] = imputer.fit_transform(self.data[self.feature_columns])
        
        # Add engineered features
        self.data['RT_GR_Ratio'] = self.data['RT'] / (self.data['GR'] + 1e-3)
        self.data['NPHI_RHOB_Crossplot'] = self.data['NPHI'] * self.data['RHOB']
        
        return self.data

    def generate_targets(self):
        """Generate synthetic targets for demonstration."""
        if self.data is None:
            raise ValueError("No data loaded - run upload first")
        
        n = len(self.data)
        if n == 0:
            raise ValueError("Data is empty")
        
        # Generate synthetic targets
        np.random.seed(42)  # For reproducible results
        self.data['LITHOLOGY'] = np.random.choice(['Sandstone', 'Limestone', 'Shale'], n)
        self.data['POROSITY'] = np.clip(
            0.25 - 0.001 * self.data['GR'] + np.random.normal(0, 0.02, n), 0, 0.35)
        self.data['PERMEABILITY'] = np.clip(
            100 * np.exp(-self.data['RT'] / 50) + np.random.normal(0, 5, n), 0, 1000)
        self.data['WATER_SATURATION'] = np.clip(
            1 - self.data['POROSITY'] + np.random.normal(0, 0.05, n), 0, 1)

    def train_models(self):
        """Train all ML models with proper error handling."""
        if not HAS_SMOTE:
            raise ImportError('pip install imbalanced-learn')

        # Validate data exists
        if self.data is None:
            raise ValueError("No data loaded - run upload first")
        
        # Check required columns exist
        required_targets = ['LITHOLOGY', 'POROSITY', 'PERMEABILITY', 'WATER_SATURATION']
        missing_targets = [col for col in required_targets if col not in self.data.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}. Run 'Generate Targets' first.")

        # Check for empty data
        if len(self.data) == 0:
            raise ValueError("Data is empty")

        # Prepare features and targets
        all_features = self.feature_columns + self.extra_features
        X = self.data[all_features].copy()
        
        # Check for NaN values
        if X.isnull().any().any():
            raise ValueError("Features contain NaN values after preprocessing")
        
        # Encode lithology
        y_lith = self.encoder.fit_transform(self.data['LITHOLOGY'])
        y_por = self.data['POROSITY'].values
        y_perm = self.data['PERMEABILITY'].values
        y_sw = self.data['WATER_SATURATION'].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Apply SMOTE for class balancing
        try:
            X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_scaled, y_lith)
        except Exception as e:
            raise ValueError(f"SMOTE resampling failed: {str(e)}")

        # Train lithology classifier
        if HAS_XGB:
            base_clf = XGBClassifier(
                objective='multi:softprob',
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.03, 0.1]
            }
        else:
            base_clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10]
            }

        # Grid search for best parameters
        grid_search = GridSearchCV(
            base_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        self.lithology_model = grid_search.fit(X_bal, y_bal).best_estimator_

        # Calculate cross-validation accuracy
        cv_scores = cross_val_score(
            self.lithology_model, X_bal, y_bal, cv=5, scoring='accuracy'
        )
        cv_acc = cv_scores.mean()

        # Train regression models
        reg_params = {'n_estimators': 200, 'random_state': 42}
        if HAS_XGB:
            reg_params = {'n_estimators': 200, 'learning_rate': 0.03, 'random_state': 42}
            RegClass = XGBRegressor
        else:
            RegClass = RandomForestRegressor

        self.porosity_model = RegClass(**reg_params).fit(X_scaled, y_por)
        self.permeability_model = RegClass(**reg_params).fit(X_scaled, y_perm)
        self.saturation_model = RegClass(**reg_params).fit(X_scaled, y_sw)

        # Store metrics
        self.metrics = {'Lithology Accuracy (CV)': float(cv_acc)}

    def make_plot(self):
        """Create professional 5-track well log plot."""
        if self.data is None:
            raise ValueError("No data loaded")

        if 'DEPTH' not in self.data.columns:
            raise ValueError("DEPTH column missing from data")

        depth = self.data['DEPTH']
        fig, axes = plt.subplots(1, 5, figsize=(14, 8), sharey=True)

        # Track 1: Gamma Ray
        axes[0].plot(self.data['GR'], depth, 'g-', linewidth=0.8)
        axes[0].set_xlim(0, 200)
        axes[0].set_xlabel('GR (API)')
        axes[0].set_title('Gamma Ray')

        # Track 2: Resistivity (log scale)
        axes[1].semilogx(self.data['RT'], depth, 'r-', linewidth=0.8)
        axes[1].set_xlim(0.2, 2000)
        axes[1].set_xlabel('RT (Ω·m)')
        axes[1].set_title('Resistivity')

        # Track 3: Neutron Porosity
        axes[2].plot(self.data['NPHI'], depth, 'b-', linewidth=0.8)
        axes[2].set_xlim(0, 0.5)
        axes[2].set_xlabel('NPHI')
        axes[2].set_title('Neutron Porosity')

        # Track 4: Bulk Density
        axes[3].plot(self.data['RHOB'], depth, 'k-', linewidth=0.8)
        axes[3].set_xlim(1.95, 2.95)
        axes[3].set_xlabel('RHOB (g/cc)')
        axes[3].set_title('Bulk Density')

        # Track 5: Lithology
        lith_colors = {
            'Sandstone': 'gold',
            'Limestone': 'skyblue', 
            'Shale': 'brown'
        }
        
        if 'LITHOLOGY' in self.data.columns:
            for i in range(len(depth) - 1):
                lith = self.data['LITHOLOGY'].iloc[i]
                color = lith_colors.get(lith, 'grey')
                axes[4].fill_betweenx(
                    [depth.iloc[i], depth.iloc[i+1]], 0, 1,
                    color=color, alpha=0.7
                )
        
        axes[4].set_xlim(0, 1)
        axes[4].set_xticks([])
        axes[4].set_title('Lithology')

        # Apply common formatting
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Depth (ft)')

        fig.tight_layout()
        return fig

    def generate_recommendations(self):
        """Generate reservoir recommendations based on trained models."""
        if self.scaler is None:
            raise RuntimeError("Models not trained - run 'Train Models' first")

        # Get features and make predictions
        X = self.data[self.feature_columns + self.extra_features]
        X_scaled = self.scaler.transform(X)

        lith_pred = self.encoder.inverse_transform(self.lithology_model.predict(X_scaled))
        por_pred = self.porosity_model.predict(X_scaled)
        perm_pred = self.permeability_model.predict(X_scaled)
        sw_pred = self.saturation_model.predict(X_scaled)

        # Create results dataframe
        results = pd.DataFrame({
            'DEPTH': self.data['DEPTH'],
            'LITH': lith_pred,
            'PHI': por_pred,
            'PERM': perm_pred,
            'SW': sw_pred
        })

        # Define quality zones
        pay_zone = (results['PHI'] > 0.20) & (results['PERM'] > 100) & (results['SW'] < 0.6)
        water_zone = results['SW'] > 0.70
        frac_zone = results['PERM'].between(50, 100)
        sand_zone = results['LITH'] == 'Sandstone'

        # Helper function for depth ranges
        def depth_range(mask):
            if mask.any():
                return f"{results[mask]['DEPTH'].min():.1f}-{results[mask]['DEPTH'].max():.1f} ft"
            return "-"

        # Build recommendation message
        msg = "=== AI-Powered Recommendations ===\n"
        msg += f"Average Properties:\n"
        msg += f"  Porosity: {por_pred.mean():.2%}\n"
        msg += f"  Permeability: {perm_pred.mean():.1f} mD\n"
        msg += f"  Water Saturation: {sw_pred.mean():.2%}\n\n"
        msg += "Zone Analysis:\n"
        msg += f"  High-quality pay zones: {pay_zone.sum():4d} intervals @ {depth_range(pay_zone)}\n"
        msg += f"  High water risk zones:  {water_zone.sum():4d} intervals @ {depth_range(water_zone)}\n"
        msg += f"  Frac candidate zones:   {frac_zone.sum():4d} intervals @ {depth_range(frac_zone)}\n"
        msg += f"  Sandstone zones:        {sand_zone.sum():4d} intervals @ {depth_range(sand_zone)}\n"

        return msg

