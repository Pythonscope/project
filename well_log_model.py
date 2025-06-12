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
        """Create professional 5-track well log plot with lithology legend."""
        if self.data is None:
            raise ValueError("No data loaded")

        if 'DEPTH' not in self.data.columns:
            raise ValueError("DEPTH column missing from data")

        depth = self.data['DEPTH']
        fig, axes = plt.subplots(1, 5, figsize=(16, 10), sharey=True)

        # Track 1: Gamma Ray
        axes[0].plot(self.data['GR'], depth, 'g-', linewidth=0.8)
        axes[0].set_xlim(0, 200)
        axes[0].set_xlabel('GR (API)')
        axes[0].set_title('Gamma Ray')
        axes[0].grid(True, alpha=0.3)

        # Track 2: Resistivity (log scale)
        axes[1].semilogx(self.data['RT'], depth, 'r-', linewidth=0.8)
        axes[1].set_xlim(0.2, 2000)
        axes[1].set_xlabel('RT (Ω·m)')
        axes[1].set_title('Resistivity')
        axes[1].grid(True, alpha=0.3)

        # Track 3: Neutron Porosity
        axes[2].plot(self.data['NPHI'], depth, 'b-', linewidth=0.8)
        axes[2].set_xlim(0, 0.5)
        axes[2].set_xlabel('NPHI')
        axes[2].set_title('Neutron Porosity')
        axes[2].grid(True, alpha=0.3)

        # Track 4: Bulk Density
        axes[3].plot(self.data['RHOB'], depth, 'k-', linewidth=0.8)
        axes[3].set_xlim(1.95, 2.95)
        axes[3].set_xlabel('RHOB (g/cc)')
        axes[3].set_title('Bulk Density')
        axes[3].grid(True, alpha=0.3)

        # Track 5: Lithology with Enhanced Legend
        lith_colors = {
            'Sandstone': '#FFD700',    # Gold/Yellow
            'Limestone': '#87CEEB',    # Sky Blue
            'Shale': '#8B4513'         # Brown
        }
        
        if 'LITHOLOGY' in self.data.columns:
            # Fill lithology intervals
            for i in range(len(depth) - 1):
                lith = self.data['LITHOLOGY'].iloc[i]
                color = lith_colors.get(lith, 'grey')
                axes[4].fill_betweenx(
                    [depth.iloc[i], depth.iloc[i+1]], 0, 1,
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.1
                )
            
            # Create legend
            legend_elements = []
            unique_liths = self.data['LITHOLOGY'].unique()
            for lith in unique_liths:
                if lith in lith_colors:
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, 
                                    facecolor=lith_colors[lith], 
                                    alpha=0.8, 
                                    edgecolor='black',
                                    label=lith)
                    )
            
            # Add legend outside the plot area
            axes[4].legend(handles=legend_elements, 
                          loc='center left', 
                          bbox_to_anchor=(1.05, 0.5),
                          fontsize=10,
                          title='Lithology')
        
        axes[4].set_xlim(0, 1)
        axes[4].set_xticks([])
        axes[4].set_title('Lithology')
        axes[4].grid(True, alpha=0.3)

        # Apply common formatting to all tracks
        for i, ax in enumerate(axes):
            ax.invert_yaxis()
            ax.set_ylabel('Depth (ft)' if i == 0 else '')
            
            # Add depth tick marks on the right side for better readability
            if i == len(axes) - 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
                ax.set_ylabel('Depth (ft)')

        # Add overall title
        fig.suptitle('AI Well Log Interpretation - Professional Analysis', fontsize=14, fontweight='bold')
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)
        
        return fig

    def generate_recommendations(self):
        """Generate comprehensive reservoir recommendations with proper formatting."""
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
            'SW': sw_pred,
            'GR': self.data['GR'],
            'RT': self.data['RT']
        })

        # Calculate hydrocarbon saturation
        results['SH'] = 1 - results['SW']

        # Define comprehensive quality zones
        pay_zone = (results['PHI'] > 0.12) & (results['PERM'] > 10) & (results['SW'] < 0.65) & (results['LITH'] == 'Sandstone')
        excellent_pay = (results['PHI'] > 0.20) & (results['PERM'] > 100) & (results['SW'] < 0.50)
        good_pay = (results['PHI'] > 0.15) & (results['PERM'] > 50) & (results['SW'] < 0.60)
        tight_gas_candidate = (results['PHI'] > 0.08) & (results['PERM'] < 10) & (results['SW'] < 0.70) & (results['LITH'] == 'Sandstone')
        water_zone = results['SW'] > 0.80
        transition_zone = results['SW'].between(0.65, 0.80)
        shale_zone = results['LITH'] == 'Shale'
        clean_sand = (results['LITH'] == 'Sandstone') & (results['GR'] < 60)
        frac_candidates = (results['PERM'].between(1, 50)) & (results['PHI'] > 0.08) & (results['LITH'] == 'Sandstone')

        # Helper functions
        def depth_range(mask):
            if mask.any():
                depths = results[mask]['DEPTH']
                return f"{depths.min():.0f}-{depths.max():.0f} ft"
            return "None identified"

        def zone_thickness(mask):
            if mask.any():
                return f"{mask.sum()} ft"
            return "0 ft"

        def avg_properties(mask, prop):
            if mask.any():
                return results[mask][prop].mean()
            return 0

        # Build comprehensive recommendation report with proper line formatting
        msg = "═══════════════════════════════════════════════════════════════\n"
        msg += "                    AI-POWERED RESERVOIR ANALYSIS               \n"
        msg += "═══════════════════════════════════════════════════════════════\n"
        msg += "\n"

        # EXECUTIVE SUMMARY
        msg += "🎯 EXECUTIVE SUMMARY\n"
        msg += "─" * 50 + "\n"
        msg += f"• Total analyzed interval: {results['DEPTH'].min():.0f} - {results['DEPTH'].max():.0f} ft ({len(results)} ft)\n"
        msg += f"• Primary lithology: {results['LITH'].mode()[0]} ({results['LITH'].value_counts().iloc[0]} ft)\n"
        overall_quality = 'Excellent' if excellent_pay.sum() > len(results)*0.3 else 'Good' if good_pay.sum() > len(results)*0.2 else 'Fair' if pay_zone.sum() > len(results)*0.1 else 'Poor'
        msg += f"• Overall reservoir quality: {overall_quality}\n"
        msg += f"• Net-to-gross ratio: {pay_zone.sum()/len(results):.1%}\n"
        msg += "\n"

        # PETROPHYSICAL PROPERTIES
        msg += "📊 AVERAGE PETROPHYSICAL PROPERTIES\n"
        msg += "─" * 50 + "\n"
        msg += f"• Porosity (PHIE):        {por_pred.mean():.1%}\n"
        msg += f"  └─ Range: {por_pred.min():.1%} - {por_pred.max():.1%}\n"
        msg += f"• Permeability (k):       {perm_pred.mean():.1f} mD\n"
        msg += f"  └─ Range: {perm_pred.min():.1f} - {perm_pred.max():.1f} mD\n"
        msg += f"• Water Saturation (Sw):  {sw_pred.mean():.1%}\n"
        msg += f"  └─ Range: {sw_pred.min():.1%} - {sw_pred.max():.1%}\n"
        msg += f"• Hydrocarbon Sat. (Sh):  {(1-sw_pred).mean():.1%}\n"
        msg += f"  └─ Range: {(1-sw_pred).min():.1%} - {(1-sw_pred).max():.1%}\n"
        msg += "\n"

        # RESERVOIR ZONATION
        msg += "🗂️  RESERVOIR ZONATION & QUALITY ASSESSMENT\n"
        msg += "─" * 50 + "\n"
        msg += f"• Excellent Pay Zones:    {zone_thickness(excellent_pay)}\n"
        msg += f"  └─ Depth: {depth_range(excellent_pay)}\n"
        if excellent_pay.any():
            msg += f"  └─ Avg. Porosity: {avg_properties(excellent_pay, 'PHI'):.1%}\n"
            msg += f"  └─ Avg. Perm: {avg_properties(excellent_pay, 'PERM'):.0f} mD\n"
            msg += f"  └─ Avg. Sw: {avg_properties(excellent_pay, 'SW'):.1%}\n"
        msg += "\n"
        
        msg += f"• Good Pay Zones:         {zone_thickness(good_pay)}\n"
        msg += f"  └─ Depth: {depth_range(good_pay)}\n"
        if good_pay.any():
            msg += f"  └─ Avg. Porosity: {avg_properties(good_pay, 'PHI'):.1%}\n"
            msg += f"  └─ Avg. Perm: {avg_properties(good_pay, 'PERM'):.0f} mD\n"
            msg += f"  └─ Avg. Sw: {avg_properties(good_pay, 'SW'):.1%}\n"
        msg += "\n"
        
        msg += f"• Marginal Pay Zones:     {zone_thickness(pay_zone & ~good_pay)}\n"
        msg += f"  └─ Depth: {depth_range(pay_zone & ~good_pay)}\n"
        msg += f"• Tight Gas Candidates:   {zone_thickness(tight_gas_candidate)}\n"
        msg += f"  └─ Depth: {depth_range(tight_gas_candidate)}\n"
        msg += f"• Transition Zones:       {zone_thickness(transition_zone)}\n"
        msg += f"  └─ Depth: {depth_range(transition_zone)}\n"
        msg += f"• Water Zones:            {zone_thickness(water_zone)}\n"
        msg += f"  └─ Depth: {depth_range(water_zone)}\n"
        msg += "\n"

        # LITHOLOGICAL ANALYSIS
        msg += "🪨 LITHOLOGICAL DISTRIBUTION\n"
        msg += "─" * 50 + "\n"
        for lith, count in results['LITH'].value_counts().items():
            percentage = count / len(results) * 100
            msg += f"• {lith:12s}: {count:3d} ft ({percentage:4.1f}%)\n"
            msg += f"  └─ Depth: {depth_range(results['LITH'] == lith)}\n"
        msg += f"• Clean Sandstone:        {zone_thickness(clean_sand)}\n"
        msg += f"  └─ Depth: {depth_range(clean_sand)}\n"
        msg += "\n"

        # COMPLETION RECOMMENDATIONS
        msg += "🔧 COMPLETION & DEVELOPMENT RECOMMENDATIONS\n"
        msg += "─" * 50 + "\n"
        
        if excellent_pay.sum() > 10:
            msg += f"✅ PRIMARY TARGETS ({excellent_pay.sum()} ft):\n"
            msg += f"   • Conventional completion recommended\n"
            msg += f"   • High production potential zones\n"
            msg += f"   • Consider for primary perforation intervals\n"
            msg += f"   • Depth: {depth_range(excellent_pay)}\n"
            msg += "\n"
        
        if frac_candidates.sum() > 5:
            msg += f"⚡ FRACTURING CANDIDATES ({frac_candidates.sum()} ft):\n"
            msg += f"   • Hydraulic fracturing recommended\n"
            msg += f"   • Multi-stage completion design\n"
            msg += f"   • Enhanced recovery potential\n"
            msg += f"   • Depth: {depth_range(frac_candidates)}\n"
            msg += "\n"
        
        if tight_gas_candidate.sum() > 5:
            msg += f"🎯 TIGHT GAS POTENTIAL ({tight_gas_candidate.sum()} ft):\n"
            msg += f"   • Unconventional development approach\n"
            msg += f"   • Enhanced recovery techniques required\n"
            msg += f"   • Consider horizontal drilling + multi-frac\n"
            msg += f"   • Depth: {depth_range(tight_gas_candidate)}\n"
            msg += "\n"

        # PRODUCTION FORECAST
        msg += "📈 PRODUCTION INSIGHTS\n"
        msg += "─" * 50 + "\n"
        total_pay = pay_zone.sum()
        if total_pay > 20:
            msg += f"🟢 PRODUCTION OUTLOOK: Favorable ({total_pay} ft net pay)\n"
            msg += f"   • Expected production: Good to excellent\n"
            drive_mech = 'Solution gas' if sw_pred.mean() < 0.6 else 'Water drive'
            msg += f"   • Primary drive mechanism: {drive_mech}\n"
        elif total_pay > 10:
            msg += f"🟡 PRODUCTION OUTLOOK: Moderate ({total_pay} ft net pay)\n"
            msg += f"   • Expected production: Fair to good\n"
            msg += f"   • Enhanced recovery may be required\n"
        else:
            msg += f"🔴 PRODUCTION OUTLOOK: Challenging ({total_pay} ft net pay)\n"
            msg += f"   • Expected production: Limited\n"
            msg += f"   • Consider alternative completion strategies\n"
        msg += "\n"

        msg += f"📊 KEY RESERVOIR METRICS:\n"
        msg += f"   • Net Pay Ratio: {total_pay/len(results):.1%}\n"
        estimated_pressure = 1000 + results['DEPTH'].mean() * 0.43
        msg += f"   • Est. Reservoir Pressure: {estimated_pressure:.0f} psi\n"
        completion_stages = max(1, frac_candidates.sum()//10)
        msg += f"   • Recommended Completion Stages: {completion_stages}\n"
        msg += "\n"

        # DRILLING RECOMMENDATIONS
        msg += "🚧 DRILLING & LOGGING RECOMMENDATIONS\n"
        msg += "─" * 50 + "\n"
        
        if water_zone.sum() > len(results) * 0.3:
            msg += f"⚠️  HIGH WATER RISK:\n"
            msg += f"   • Monitor water production closely\n"
            msg += f"   • Consider water shut-off techniques\n"
        
        if shale_zone.sum() > len(results) * 0.4:
            msg += f"⚠️  SHALE DOMINANT SEQUENCE:\n"
            msg += f"   • Consider shale gas potential\n"
            msg += f"   • Evaluate horizontal drilling options\n"
        
        if results['PHI'].std() > 0.05:
            msg += f"📈 HIGH RESERVOIR HETEROGENEITY:\n"
            msg += f"   • Detailed reservoir modeling recommended\n"
            msg += f"   • Consider additional log acquisition\n"
        
        msg += f"🔍 ADDITIONAL RECOMMENDATIONS:\n"
        msg += f"   • Advanced logging for detailed characterization\n"
        if water_zone.any():
            msg += f"   • Set casing above {results[water_zone]['DEPTH'].min():.0f} ft if possible\n"
        msg += f"   • Consider pressure testing in pay zones\n"
        msg += f"   • Implement real-time drilling optimization\n"
        msg += "\n"

        msg += "═══════════════════════════════════════════════════════════════\n"
        msg += "                         END OF ANALYSIS                        \n"
        msg += "═══════════════════════════════════════════════════════════════"

        return msg
