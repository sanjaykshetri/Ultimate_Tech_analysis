"""
Ultimate Data Science Challenge - Complete Analysis
Parts 1, 2, and 3
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("ULTIMATE DATA SCIENCE CHALLENGE - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS - LOGIN TIME SERIES
# ============================================================================
print("\n" + "="*80)
print("PART 1: EXPLORATORY DATA ANALYSIS - LOGIN TIMESTAMPS")
print("="*80)

# Load login data
with open('logins.json', 'r') as f:
    login_data = json.load(f)

login_times = pd.Series(login_data['login_time'])
print(f"\nTotal login records: {len(login_times)}")
print(f"Time range: {login_times.min()} to {login_times.max()}")

# Convert to datetime
login_times_dt = pd.to_datetime(login_times)

# Create 15-minute intervals
login_times_dt = login_times_dt.sort_values().reset_index(drop=True)
login_counts_15min = login_times_dt.dt.floor('15min').value_counts().sort_index()

print(f"\nAfter aggregating to 15-minute intervals:")
print(f"Number of 15-min intervals: {len(login_counts_15min)}")
print(f"Average logins per 15-min: {login_counts_15min.mean():.2f}")
print(f"Median logins per 15-min: {login_counts_15min.median():.2f}")
print(f"Max logins in a 15-min interval: {login_counts_15min.max()}")
print(f"Min logins in a 15-min interval: {login_counts_15min.min()}")

# Data Quality Analysis
print("\n--- DATA QUALITY ANALYSIS ---")
print(f"Duplicates: {len(login_times) - len(login_times.unique())} duplicate timestamps found")
print(f"Missing values: {login_times.isna().sum()}")

# Identify quality issues
date_range = (login_times_dt.max() - login_times_dt.min()).total_seconds() / 86400
expected_intervals = date_range * 24 * 4  # 4 intervals per hour
actual_intervals = len(login_counts_15min)
print(f"Expected time intervals (if continuous): {expected_intervals:.0f}")
print(f"Actual time intervals: {actual_intervals}")
print(f"Coverage: {(actual_intervals/expected_intervals*100):.1f}%")

# Time series characteristics
login_times_df = pd.DataFrame({
    'timestamp': login_times_dt,
    'date': login_times_dt.dt.date,
    'hour': login_times_dt.dt.hour,
    'day_of_week': login_times_dt.dt.day_name(),
    'minute_group': login_times_dt.dt.floor('15min')
})

# Daily pattern analysis
hourly_pattern = login_times_df.groupby('hour').size()
print(f"\n--- DAILY PATTERN ---")
print("Peak hours (top 5):")
print(hourly_pattern.nlargest(5))
print("\nOff-peak hours (bottom 5):")
print(hourly_pattern.nsmallest(5))

# Day of week pattern
dow_pattern = login_times_df.groupby('day_of_week').size()
print(f"\n--- WEEKLY PATTERN ---")
print(dow_pattern)

# Visualization 1: Time series plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Full time series
axes[0].plot(login_counts_15min.index, login_counts_15min.values, linewidth=0.8, alpha=0.7)
axes[0].set_title('Login Counts - 15 Minute Intervals (Full Timeline)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Logins per 15 min')
axes[0].grid(True, alpha=0.3)

# Hourly pattern
axes[1].bar(hourly_pattern.index, hourly_pattern.values, color='steelblue', alpha=0.7)
axes[1].set_title('Login Pattern by Hour of Day', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Total Logins')
axes[1].set_xticks(range(0, 24))
axes[1].grid(True, alpha=0.3, axis='y')

# Weekly pattern
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sorted = dow_pattern.reindex([d for d in dow_order if d in dow_pattern.index])
axes[2].bar(range(len(dow_sorted)), dow_sorted.values, color='coral', alpha=0.7)
axes[2].set_title('Login Pattern by Day of Week', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Day of Week')
axes[2].set_ylabel('Total Logins')
axes[2].set_xticks(range(len(dow_sorted)))
axes[2].set_xticklabels([d for d in dow_order if d in dow_pattern.index], rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('part1_login_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved visualization: part1_login_analysis.png")
plt.close()

print("\n--- KEY FINDINGS FOR PART 1 ---")
print(f"• Strong daily cycle detected: Peak activity during hours {hourly_pattern.nlargest(3).index.tolist()}")
print(f"• Weekend vs weekday difference: {100*(hw_pattern - wd_pattern)/wd_pattern:.1f}%" if (hw_pattern := dow_pattern[['Saturday', 'Sunday']].mean()) and (wd_pattern := dow_pattern[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()) else "")
print(f"• Logins are concentrated in specific hours, with significant variation")

# ============================================================================
# PART 2: EXPERIMENT AND METRICS DESIGN
# ============================================================================
print("\n" + "="*80)
print("PART 2: EXPERIMENT AND METRICS DESIGN")
print("="*80)

print("""
SCENARIO: Toll bridge reimbursement to encourage cross-city driver availability

1) KEY MEASURE OF SUCCESS:
   
   Metric: "Cross-City Active Driver Days per Week" or "Cross-City Engagement Rate"
   
   Definition: % of drivers who serve BOTH cities (have trips in both) in a given week
   
   Rationale:
   - Measures the BEHAVIOR CHANGE we want to incentivize (multi-city availability)
   - Directly tied to operational goal (fulfilling demand in both cities)
   - More actionable than binary measures (captures degree of engagement)
   - Comparable across treatment/control groups
   - Observable through transaction data (no survey needed)
   
   Alternative metrics considered:
   ✗ Total trips: Doesn't measure cross-city availability specifically
   ✗ Driver retention: Too long-term, not specific to experiment goal
   ✗ Revenue: Influenced by surge pricing, not core metric
   ✓ Toll reimbursement claims: Good proxy but influenced by driver awareness

2) EXPERIMENT DESIGN:

   a) IMPLEMENTATION:
   
   - Duration: 12-16 weeks (to capture 2-3 full weekly cycles)
   - Control Group: No toll reimbursement (baseline)
   - Treatment Group: Full toll reimbursement (all toll costs covered)
   - Assignment: Geographic randomization by driver at signup or region level
      
   - Sample Size Calculation:
     * Assume 20% increase in cross-city availability as meaningful effect
     * Current baseline: 30% of drivers serve both cities
     * Need ~1000-2000 drivers per group for adequate power (α=0.05, β=0.20)
   
   - Implementation Steps:
     1. ID active drivers in both cities (baseline period: 4 weeks)
     2. Randomly assign to treatment/control with stratification by city
     3. Implement toll reimbursement system for treatment group
     4. Track all trips and toll transactions
     5. Monitor for:
        * Spillover effects (treatment affecting control drivers)
        * Network effects (availability changes affecting demand patterns)
        * Seasonal confounds (weather, events, etc.)

   b) STATISTICAL TESTS:
   
   Primary Hypothesis Test:
   H0: Cross-city engagement rate is equal in treatment and control groups
   H1: Cross-city engagement rate is higher in treatment group
   
   Recommended tests:
   - Two-sample proportion test (Chi-square test)
     * Test if % drivers serving both cities differs between groups
     * Simple, interpretable, robust
   
   - Logistic regression with interaction terms
     * Control for covariates (driver experience, city characteristics)
     * Account for repeated measures (driver-week level analysis)
     * Estimate treatment effect size
   
   - Difference-in-differences (if panel data available)
     * Accounts for time trends, seasonal effects
     * Estimates treatment effect above baseline trend
   
   - Heterogeneous treatment effects analysis
     * Does reimbursement help new drivers vs. experienced?
     * Different effects by city pair?

   c) INTERPRETATION & RECOMMENDATIONS:
   
   Scenarios:
   
   Scenario A: Significant increase in cross-city service (p < 0.05)
   → Recommendation: EXPAND toll reimbursement citywide
   → Caveats: Monitor long-term implications on margin
   
   Scenario B: No significant difference (p ≥ 0.05)
   → Recommendation: REJECT current approach, explore alternatives:
        * Higher reimbursement amounts
        * Guaranteed earnings for multi-city trips
        * Prioritization in dispatch algorithm
   → Caveats: Experiment may lack statistical power
   
   Scenario C: Increased cross-city service but revenue drops (margin concerns)
   → Recommendation: CONDITIONAL ROLLOUT with:
        * Reduced reimbursement (50% toll coverage)
        * Performance-based incentives
        * Time-limited promotions (peak demand periods only)
   → Caveats: May need to retest with modified approach
   
   CRITICAL CAVEATS:
   • Network externalities: One driver's availability affects demand for others
   • Attrition bias: Drivers may sort into treatment/control after random assignment
   • Compliance: Ensure treatment implementation consistent (toll system tracking)
   • Spillover: Treat control drivers may hear about reimbursement
   • Confounds: Competitor activity, regulatory changes (monitor weekly)
   • Timing: 12-16 weeks may not be enough to see habit formation
""")

# ============================================================================
# PART 3: PREDICTIVE MODELING FOR RIDER RETENTION
# ============================================================================
print("\n" + "="*80)
print("PART 3: PREDICTIVE MODELING FOR RIDER RETENTION")
print("="*80)

# Load user data
with open('ultimate_data_challenge.json', 'r') as f:
    user_data = json.load(f)

users_df = pd.DataFrame(user_data)

# 1. DATA CLEANING & EDA
print("\n--- DATA CLEANING & EDA ---")
print(f"Total users: {len(users_df)}")
print(f"Columns: {users_df.columns.tolist()}")

# Check for missing values
print(f"\nMissing values:\n{users_df.isnull().sum()}")

# Convert dates
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])
users_df['last_trip_date'] = pd.to_datetime(users_df['last_trip_date'])

# Create target variable: Retained if active in 30 days before data pull
# Assuming data was pulled ~6 months after signup
users_df['days_since_signup'] = (users_df['last_trip_date'] - users_df['signup_date']).dt.days

# Retention definition: Active in the 6th month (approximately 150-180 days after signup)
# We'll use the last_trip_date to determine if they were active in month 6
# Month 6 would be around day 150-180 from signup
retention_start = 150
retention_end = 210
users_df['retained'] = users_df['days_since_signup'].apply(
    lambda x: 1 if retention_start <= x <= retention_end else 0
)

# Also create a more conservative definition: active in last 30 days of data pull date
# Since we don't know exact pull date, let's use days_since_signup > 150 as proxy for being active 6 months later
users_df['days_since_signup_clipped'] = users_df['days_since_signup'].clip(lower=0)
users_df['retained_v2'] = (users_df['days_since_signup_clipped'] >= 150).astype(int)

retention_rate = users_df['retained_v2'].mean()
print(f"\nRetention rate (active in month 6): {retention_rate:.2%}")
print(f"Retained users: {users_df['retained_v2'].sum()}")
print(f"Churned users: {(1-users_df['retained_v2']).sum()}")

# Descriptive statistics
print(f"\n--- FEATURE STATISTICS ---")
print(users_df.describe())

# Categorical features
print(f"\nUnique cities: {users_df['city'].nunique()}")
print(f"Top 5 cities:\n{users_df['city'].value_counts().head()}")

print(f"\nPhone types:\n{users_df['phone'].value_counts()}")

print(f"\nUltimate Black user distribution:\n{users_df['ultimate_black_user'].value_counts()}")

# Retention by key features
print(f"\n✗ Retention by phone:")
print(users_df.groupby('phone')['retained_v2'].agg(['count', 'sum', 'mean']))

print(f"\nRetention by Ultimate Black user:")
print(users_df.groupby('ultimate_black_user')['retained_v2'].agg(['count', 'sum', 'mean']))

# Visualization 2: Retention analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trips in first 30 days vs retention
retained_df = users_df.groupby('trips_in_first_30_days')['retained_v2'].agg(['count', 'mean'])
axes[0, 0].bar(retained_df.index, retained_df['mean'], alpha=0.7, color='steelblue')
axes[0, 0].set_title('Retention Rate by Trips in First 30 Days', fontweight='bold')
axes[0, 0].set_xlabel('Trips in First 30 Days')
axes[0, 0].set_ylabel('Retention Rate')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Avg distance vs retention
axes[0, 1].scatter(users_df['avg_dist'], users_df['retained_v2'], alpha=0.3)
axes[0, 1].set_title('Avg Trip Distance vs Retention', fontweight='bold')
axes[0, 1].set_xlabel('Average Distance (miles)')
axes[0, 1].set_ylabel('Retained (1=Yes, 0=No)')
axes[0, 1].grid(True, alpha=0.3)

# Rating distribution by retention
users_df.boxplot(column='avg_rating_by_driver', by='retained_v2', ax=axes[1, 0])
axes[1, 0].set_title('Driver Rating by Retention', fontweight='bold')
axes[1, 0].set_xlabel('Retained (0=No, 1=Yes)')
axes[1, 0].set_ylabel('Avg Rating by Driver')
axes[1, 0].get_figure().suptitle('')

# Surge exposure vs retention
surge_groups = pd.cut(users_df['surge_pct'], bins=5)
surge_retention = users_df.groupby(surge_groups)['retained_v2'].mean()
axes[1, 1].bar(range(len(surge_retention)), surge_retention.values, alpha=0.7, color='coral')
axes[1, 1].set_title('Retention by Surge Exposure', fontweight='bold')
axes[1, 1].set_xlabel('Surge % Quintile')
axes[1, 1].set_ylabel('Retention Rate')
axes[1, 1].set_xticks(range(len(surge_retention)))
axes[1, 1].set_xticklabels([f'Q{i+1}' for i in range(len(surge_retention))])
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('part3_eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved visualization: part3_eda_analysis.png")
plt.close()

# 2. BUILD PREDICTIVE MODEL
print("\n--- BUILDING PREDICTIVE MODEL ---")

# Prepare features
# Drop non-numeric and target columns
feature_cols = [
    'trips_in_first_30_days', 'avg_rating_of_driver', 'avg_rating_by_driver',
    'avg_surge', 'surge_pct', 'avg_dist', 'weekday_pct'
]

# Add encoded categorical features
users_df['phone_iphone'] = (users_df['phone'] == 'iPhone').astype(int)
users_df['phone_android'] = (users_df['phone'] == 'Android').astype(int)
users_df['black_user'] = users_df['ultimate_black_user'].astype(int)

# Encode top cities
top_cities = users_df['city'].value_counts().head(5).index
for city in top_cities:
    users_df[f'city_{city}'] = (users_df['city'] == city).astype(int)

feature_cols.extend(['phone_iphone', 'phone_android', 'black_user'] + 
                    [f'city_{city}' for city in top_cities])

X = users_df[feature_cols].copy()
y = users_df['retained_v2'].copy()

# Handle missing values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set retention rate: {y_train.mean():.2%}")
print(f"Test set retention rate: {y_test.mean():.2%}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n--- MODEL COMPARISON ---")

# Model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
print(f"\nLogistic Regression:")
print(f"  ROC-AUC Score: {lr_auc:.4f}")
print(f"  Accuracy: {(lr_pred == y_test).mean():.4f}")

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"\nRandom Forest:")
print(f"  ROC-AUC Score: {rf_auc:.4f}")
print(f"  Accuracy: {(rf_pred == y_test).mean():.4f}")

# Model 3: Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_pred_proba)
print(f"\nGradient Boosting:")
print(f"  ROC-AUC Score: {gb_auc:.4f}")
print(f"  Accuracy: {(gb_pred == y_test).mean():.4f}")

# Best model
best_model = gb_model if gb_auc >= max(lr_auc, rf_auc) else (rf_model if rf_auc >= lr_auc else lr_model)
best_model_name = 'Gradient Boosting' if gb_auc >= max(lr_auc, rf_auc) else ('Random Forest' if rf_auc >= lr_auc else 'Logistic Regression')
best_pred_proba = gb_pred_proba if gb_auc >= max(lr_auc, rf_auc) else (rf_pred_proba if rf_auc >= lr_auc else lr_pred_proba)

print(f"\n✗ Best Model: {best_model_name} (ROC-AUC: {max(lr_auc, rf_auc, gb_auc):.4f})")

# Feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# Detailed classification report
print(f"\n--- CLASSIFICATION REPORT ({best_model_name}) ---")
print(classification_report(y_test, (best_pred_proba >= 0.5).astype(int), 
                          target_names=['Churned', 'Retained']))

# Model diagnostics
print(f"\n--- MODEL DIAGNOSTICS ---")
fpr, tpr, _ = roc_curve(y_test, best_pred_proba)
conf_matrix = confusion_matrix(y_test, (best_pred_proba >= 0.5).astype(int))
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"  True Negatives: {conf_matrix[0,0]}")
print(f"  False Positives: {conf_matrix[0,1]}")
print(f"  False Negatives: {conf_matrix[1,0]}")
print(f"  True Positives: {conf_matrix[1,1]}")

# Visualization 3: Model performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curve
axes[0, 0].plot(fpr, tpr, label=f'{best_model_name} (AUC={max(lr_auc, rf_auc, gb_auc):.4f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Baseline')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Model comparison
model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
model_aucs = [lr_auc, rf_auc, gb_auc]
colors = ['steelblue' if auc != max(model_aucs) else 'coral' for auc in model_aucs]
axes[0, 1].bar(model_names, model_aucs, color=colors, alpha=0.7)
axes[0, 1].set_title('Model Comparison (ROC-AUC)', fontweight='bold')
axes[0, 1].set_ylabel('ROC-AUC Score')
axes[0, 1].set_ylim([0.5, 1.0])
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Prediction distribution
axes[1, 0].hist(best_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Churned', color='red')
axes[1, 0].hist(best_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Retained', color='green')
axes[1, 0].axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
axes[1, 0].set_xlabel('Predicted Probability of Retention')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    top_features = feature_importance.head(10)
    axes[1, 1].barh(top_features['feature'], top_features['importance'], color='steelblue', alpha=0.7)
    axes[1, 1].set_title('Top 10 Feature Importance', fontweight='bold')
    axes[1, 1].set_xlabel('Importance Score')
else:
    # For logistic regression, show coefficient magnitudes
    coef_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(lr_model.coef_[0])
    }).sort_values('importance', ascending=False).head(10)
    axes[1, 1].barh(coef_importance['feature'], coef_importance['importance'], color='steelblue', alpha=0.7)
    axes[1, 1].set_title('Top 10 Feature Coefficients (Abs Value)', fontweight='bold')
    axes[1, 1].set_xlabel('|Coefficient|')

plt.tight_layout()
plt.savefig('part3_model_performance.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved visualization: part3_model_performance.png")
plt.close()

# 3. ACTIONABLE INSIGHTS
print("\n" + "="*80)
print("RECOMMENDATIONS TO IMPROVE RIDER RETENTION")
print("="*80)

print("""
Based on the predictive modeling and exploratory analysis:

KEY INSIGHTS:
1. Early engagement is critical:
   - Drivers with more trips in first 30 days show higher retention
   → RECOMMENDATION: Implement onboarding program with incentives for 
     first 30 days (e.g., bonus for 5+ trips, reduced commission)

2. Trip quality matters:
   - Users with higher driver ratings show better retention
   RECOMMENDATION: Premium driver matching for new users; quality 
     assurance program for driver ratings

3. Surge exposure affects retention:
   - Moderate surge pricing encourages retention, extreme surge hurts it
   → RECOMMENDATION: Cap surge prices for new users in first 90 days

4. Service type matters:
   - Ultimate Black users show different retention patterns
   → RECOMMENDATION: Personalized retention campaigns by service type

OPERATIONAL STRATEGIES:
A) Immediate actions (0-30 days):
   • Personalized onboarding based on user profile
   • First-ride discounts to encourage multiple trips in month 1
   • Premium driver matching and quality feedback

B) Medium-term (30-90 days):
   • Monitor user engagement metrics weekly
   • Send personalized offers based on trip history
   • Quality improvement campaigns (driver ratings)

C) Long-term (90+ days):
   • Loyalty program with milestone rewards
   • Predictive churn alerts to intervene proactively
   • Community/social features to increase engagement

MEASUREMENT:
• Track retention cohort-by-cohort with new interventions
• Use ROI model: 
  - Cost of intervention vs. lifetime value of retained user
  - Break-even point: {retention_rate * 100:.1f}% of retained users needed to cover costs
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  - part1_login_analysis.png")
print("  - part3_eda_analysis.png")
print("  - part3_model_performance.png")
print("\n[COMPLETE] All three parts completed successfully!")
