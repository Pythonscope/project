import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
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

# Use non-interactive backend for headless servers
plt.switch_backend('Agg')

class WellLogInterpreter:
    def __init__(self):
        self.data = None
        self.feature_columns = ['GR', 'RT', 'NPHI', 'RHOB', 'PEF']
        self.extra_features = ['RT_GR_Ratio', 'NPHI_RHOB_Crossplot', 'Photoelectric_Factor']
        self.lithology_model = None
        self.porosity_model = None
        self.permeability_model = None
        self.saturation_model = None
        self.scaler = None
        self.lithology_encoder = LabelEncoder()
        self.metrics = {}
        
    def preprocess_data(self, file_path):
        """Load and preprocess CSV data"""
        try:
            # Read CSV file
            if isinstance(file_path, str):
                self.data = pd.read_csv(file_path)
            else:
                # Handle file object
                self.data = pd.read_csv(file_path)
            
            # Validate required columns
            missing_cols = [col for col in self.feature_columns if col not in self.data.columns]
            if missing_cols:
                # Create missing columns with synthetic data for demo
                for col in missing_cols:
                    if col == 'GR':
                        self.data[col] = np.random.uniform(20, 150, len(self.data))
                    elif col == 'RT':
                        self.data[col] = np.random.uniform(0.5, 100, len(self.data))
                    elif col == 'NPHI':
                        self.data[col] = np.random.uniform(0.05, 0.35, len(self.data))
                    elif col == 'RHOB':
                        self.data[col] = np.random.uniform(2.0, 2.8, len(self.data))
                    elif col == 'PEF':
                        self.data[col] = np.random.uniform(1.5, 5.0, len(self.data))
            
            # Ensure DEPTH column exists
            if 'DEPTH' not in self.data.columns:
                self.data['DEPTH'] = np.arange(1000, 1000 + len(self.data), 0.5)
            
            # Handle missing values using KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            self.data[self.feature_columns] = imputer.fit_transform(
                self.data[self.feature_columns]
            )
            
            # Feature engineering
            self._create_engineered_features()
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"Data preprocessing failed: {str(e)}")
    
    def _create_engineered_features(self):
        """Create engineered features for better model performance"""
        # Resistivity to Gamma Ray ratio (hydrocarbon indicator)
        self.data['RT_GR_Ratio'] = self.data['RT'] / (self.data['GR'] + 1e-3)
        
        # Neutron-Density crossplot (lithology indicator)
        self.data['NPHI_RHOB_Crossplot'] = self.data['NPHI'] * self.data['RHOB']
        
        # Enhanced photoelectric factor
        self.data['Photoelectric_Factor'] = self.data['PEF'] * self.data['RHOB']
        
        # Clay volume indicator
        self.data['Clay_Volume'] = np.clip((self.data['GR'] - 25) / 125, 0, 1)
        
    def generate_targets(self):
        """Generate synthetic target variables for demonstration"""
        n = len(self.data)
        np.random.seed(42)  # For reproducible results
        
        # Lithology based on GR and PEF
        conditions = [
            (self.data['GR'] < 75) & (self.data['PEF'] < 2.5),  # Sandstone
            (self.data['GR'] < 75) & (self.data['PEF'] >= 2.5),  # Limestone
            self.data['GR'] >= 75  # Shale
        ]
        choices = ['Sandstone', 'Limestone', 'Shale']
        self.data['LITHOLOGY'] = np.select(conditions, choices, default='Shale')
        
        # Porosity (inversely related to density, positively to neutron)
        self.data['POROSITY'] = np.clip(
            0.45 - 0.15 * self.data['RHOB'] + 0.3 * self.data['NPHI'] + 
            np.random.normal(0, 0.02, n), 0, 0.4
        )
        
        # Permeability (related to porosity and lithology)
        base_perm = 1000 * (self.data['POROSITY'] ** 3) / ((1 - self.data['POROSITY']) ** 2)
        lithology_factor = np.where(self.data['LITHOLOGY'] == 'Sandstone', 1.5,
                          np.where(self.data['LITHOLOGY'] == 'Limestone', 0.8, 0.1))
        self.data['PERMEABILITY'] = np.clip(
            base_perm * lithology_factor + np.random.normal(0, 10, n), 
            0.01, 5000
        )
        
        # Water saturation (Archie's equation approximation)
        self.data['WATER_SATURATION'] = np.clip(
            (1 / self.data['RT']) ** 0.5 + np.random.normal(0, 0.05, n),
            0.1, 1.0
        )
    
    def train_models(self):
        """Train all ML models"""
        if self.data is None:
            raise ValueError("No data loaded. Call preprocess_data() first.")
        
        # Prepare features
        X = self.data[self.feature_columns + self.extra_features]
        
        # Prepare targets
        y_lith = self.lithology_encoder.fit_transform(self.data['LITHOLOGY'])
        y_por = self.data['POROSITY']
        y_perm = np.log10(self.data['PERMEABILITY'] + 1)  # Log transform for better distribution
        y_sw = self.data['WATER_SATURATION']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle class imbalance for lithology if SMOTE is available
        if HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y_lith)
        else:
            X_balanced, y_balanced = X_scaled, y_lith
        
        # Train lithology classifier
        self._train_lithology_model(X_balanced, y_balanced)
        
        # Train regression models
        self._train_regression_models(X_scaled, y_por, y_perm, y_sw)
        
        # Calculate metrics
        self._calculate_metrics(X_scaled, y_lith, y_por, y_perm, y_sw)
    
    def _train_lithology_model(self, X, y):
        """Train lithology classification model"""
        if HAS_XGB:
            base_model = XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.03, 0.1]
            }
        else:
            base_model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None]
            }
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        self.lithology_model = grid_search.best_estimator_
    
    def _train_regression_models(self, X, y_por, y_perm, y_sw):
        """Train regression models for petrophysical properties"""
        if HAS_XGB:
            regressor_class = XGBRegressor
            params = {'n_estimators': 200, 'learning_rate': 0.03, 'random_state': 42}
        else:
            regressor_class = RandomForestRegressor
            params = {'n_estimators': 200, 'random_state': 42}
        
        self.porosity_model = regressor_class(**params).fit(X, y_por)
        self.permeability_model = regressor_class(**params).fit(X, y_perm)
        self.saturation_model = regressor_class(**params).fit(X, y_sw)
    
    def _calculate_metrics(self, X, y_lith, y_por, y_perm, y_sw):
        """Calculate model performance metrics"""
        # Lithology accuracy
        lith_accuracy = cross_val_score(
            self.lithology_model, X, y_lith, cv=5, scoring='accuracy'
        ).mean()
        
        # Regression R² scores
        por_r2 = self.porosity_model.score(X, y_por)
        perm_r2 = self.permeability_model.score(X, y_perm)
        sw_r2 = self.saturation_model.score(X, y_sw)
        
        self.metrics = {
            'Lithology_Accuracy': float(lith_accuracy),
            'Porosity_R2': float(por_r2),
            'Permeability_R2': float(perm_r2),
            'Water_Saturation_R2': float(sw_r2)
        }
    
    def generate_recommendations(self):
        """Generate AI-powered drilling and completion recommendations"""
        if self.lithology_model is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Make predictions
        X = self.scaler.transform(self.data[self.feature_columns + self.extra_features])
        
        lith_pred = self.lithology_encoder.inverse_transform(
            self.lithology_model.predict(X)
        )
        por_pred = self.porosity_model.predict(X)
        perm_pred = np.power(10, self.permeability_model.predict(X)) - 1  # Reverse log transform
        sw_pred = self.saturation_model.predict(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'DEPTH': self.data['DEPTH'],
            'LITHOLOGY': lith_pred,
            'POROSITY': por_pred,
            'PERMEABILITY': perm_pred,
            'WATER_SATURATION': sw_pred
        })
        
        # Identify zones of interest
        pay_zones = (results['POROSITY'] > 0.15) & \
                   (results['PERMEABILITY'] > 50) & \
                   (results['WATER_SATURATION'] < 0.6)
        
        water_zones = results['WATER_SATURATION'] > 0.8
        frac_candidates = (results['PERMEABILITY'].between(10, 100)) & \
                         (results['LITHOLOGY'] == 'Sandstone')
        
        # Generate recommendations text
        recommendations = self._format_recommendations(results, pay_zones, water_zones, frac_candidates)
        
        return recommendations
    
    def _format_recommendations(self, results, pay_zones, water_zones, frac_candidates):
        """Format recommendations as readable text"""
        def depth_range(mask):
            if mask.any():
                depths = results.loc[mask, 'DEPTH']
                return f"{depths.min():.1f} - {depths.max():.1f} ft"
            return "None identified"
        
        avg_por = results['POROSITY'].mean()
        avg_perm = results['PERMEABILITY'].mean()
        avg_sw = results['WATER_SATURATION'].mean()
        
        recommendations = f"""
=== AI-POWERED WELL LOG INTERPRETATION ===

RESERVOIR QUALITY ASSESSMENT:
• Average Porosity: {avg_por:.1%}
• Average Permeability: {avg_perm:.1f} mD
• Average Water Saturation: {avg_sw:.1%}

ZONE IDENTIFICATION:
• High-Quality Pay Zones: {pay_zones.sum()} intervals at {depth_range(pay_zones)}
• Water-Bearing Zones: {water_zones.sum()} intervals at {depth_range(water_zones)}
• Hydraulic Fracturing Candidates: {frac_candidates.sum()} intervals at {depth_range(frac_candidates)}

DRILLING RECOMMENDATIONS:
• Primary Target: Focus on pay zones with Φ > 15%, k > 50 mD, Sw < 60%
• Completion Strategy: Consider multi-stage fracturing in identified candidate zones
• Water Management: Monitor water production in high Sw intervals

LITHOLOGY DISTRIBUTION:
"""
        
        # Add lithology statistics
        lith_counts = results['LITHOLOGY'].value_counts()
        for lith, count in lith_counts.items():
            percentage = (count / len(results)) * 100
            recommendations += f"• {lith}: {count} intervals ({percentage:.1f}%)\n"
        
        recommendations += "\n=== END OF REPORT ==="
        
        return recommendations
    
    def make_plot(self):
        """Create professional well log plot"""
        if self.data is None:
            raise ValueError("No data loaded. Call preprocess_data() first.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 6, figsize=(18, 10), sharey=True)
        
        depth = self.data['DEPTH']
        
        # Track 1: Gamma Ray
        axes[0].plot(self.data['GR'], depth, 'green', linewidth=1.5)
        axes[0].fill_betweenx(depth, 0, self.data['GR'], 
                             where=(self.data['GR'] > 75), alpha=0.3, color='brown', label='Shale')
        axes[0].set_xlabel('Gamma Ray\n(API)')
        axes[0].set_xlim(0, 200)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('GR')
        
        # Track 2: Resistivity (log scale)
        axes[1].semilogx(self.data['RT'], depth, 'red', linewidth=1.5)
        axes[1].set_xlabel('Resistivity\n(Ω·m)')
        axes[1].set_xlim(0.1, 1000)
        axes[1].grid(True, which='both', alpha=0.3)
        axes[1].set_title('RT')
        
        # Track 3: Neutron Porosity
        axes[2].plot(self.data['NPHI'], depth, 'blue', linewidth=1.5)
        axes[2].set_xlabel('Neutron\nPorosity')
        axes[2].set_xlim(0, 0.5)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('NPHI')
        axes[2].invert_xaxis()  # Convention for neutron logs
        
        # Track 4: Bulk Density
        axes[3].plot(self.data['RHOB'], depth, 'black', linewidth=1.5)
        axes[3].set_xlabel('Bulk Density\n(g/cc)')
        axes[3].set_xlim(1.8, 3.0)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('RHOB')
        
        # Track 5: Photoelectric Factor
        axes[4].plot(self.data['PEF'], depth, 'purple', linewidth=1.5)
        axes[4].set_xlabel('Photoelectric\nFactor')
        axes[4].set_xlim(0, 6)
        axes[4].grid(True, alpha=0.3)
        axes[4].set_title('PEF')
        
        # Track 6: Lithology (if available)
        if 'LITHOLOGY' in self.data.columns:
            lith_colors = {'Sandstone': 'gold', 'Limestone': 'lightblue', 'Shale': 'brown'}
            unique_depths = self.data['DEPTH'].values
            
            for i in range(len(unique_depths) - 1):
                lith = self.data['LITHOLOGY'].iloc[i]
                color = lith_colors.get(lith, 'gray')
                axes[5].fill_between([0, 1], unique_depths[i], unique_depths[i+1], 
                                   color=color, alpha=0.7)
            
            # Add legend
            legend_elements = [patches.Patch(color=color, label=lith) 
                             for lith, color in lith_colors.items()]
            axes[5].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        axes[5].set_xlim(0, 1)
        axes[5].set_xticks([])
        axes[5].set_title('Lithology')
        axes[5].grid(False)
        
        # Shared formatting
        for ax in axes:
            ax.set_ylabel('Depth (ft)')
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Overall title
        fig.suptitle('AI Well Log Interpretation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
