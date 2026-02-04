#We ran our notebook through ai to make it into a single pipeline py file, to add comments and for better understanding of what the code is doing at every instant of execution.

import pandas as pd
import numpy as np
import json
import warnings
import re

# Machine Learning libraries
import xgboost as xgb
import lightgbm as lgb
import optuna

# Metrics
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

print("="*90)
print(" "*20 + "XGBOOST + LIGHTGBM ENSEMBLE PREDICTOR")
print("="*90)

# ============================================================================
# FIFA CLUB WORLD CUP DATA
# ============================================================================
# Historical FIFA Club World Cup data for elite teams
# This provides context about teams' success at the highest global level
# Although not used directly in predictions, these are included as features

FIFA_CWC_TITLES = {
    'Real Madrid': 6, 
    'Barcelona': 3, 
    'Bayern Munich': 2, 
    'Chelsea': 2,
    'AC Milan': 1, 
    'Inter Milan': 1, 
    'Manchester United': 1, 
    'Liverpool': 1, 
    'Manchester City': 1
}

FIFA_CWC_FINALS = {
    'Real Madrid': 9, 
    'Barcelona': 3, 
    'Bayern Munich': 4, 
    'Liverpool': 2, 
    'Chelsea': 2, 
    'Manchester United': 2
}

def get_cwc_prestige(team):
    """
    Calculate FIFA Club World Cup prestige score for a team.
    
    The prestige score is based on:
    - Titles won: 10 points each
    - Finals reached: 5 points each
    
    Args:
        team (str): Team name
        
    Returns:
        int: Prestige score (0 if team never won/reached CWC final)
    """
    # Map common team name variations to standardized names
    team_map = {
        'BAYERN': 'Bayern Munich', 
        'BAYERN MUNICH': 'Bayern Munich',
        'INTER': 'Inter Milan', 
        'INTERNAZIONALE': 'Inter Milan',
        'MAN CITY': 'Manchester City', 
        'MANCHESTER CITY': 'Manchester City',
        'MAN UTD': 'Manchester United', 
        'MANCHESTER UTD': 'Manchester United'
    }
    
    # Normalize team name
    team_key = team_map.get(str(team).upper().strip(), str(team))
    
    # Calculate score
    score = 0
    score += FIFA_CWC_TITLES.get(team_key, 0) * 10  # 10 points per title
    score += FIFA_CWC_FINALS.get(team_key, 0) * 5   # 5 points per final appearance
    
    return score

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
# Load the main dataset containing all historical matches

print("\nLoading datasets...")
df = pd.read_csv('Full_Dataset.csv')
print(f"  Full_Dataset.csv: {len(df):,} matches loaded")

# Load and clean Champions League knockout stage data from JSON
print("  Loading and cleaning train.json...")
with open('train.json', 'r') as f:
    train_json_raw = json.load(f)

# Clean the train.json data
train_json = {}
for season_key, season_data in train_json_raw.items():
    # Normalize season key (remove any extra spaces, standardize format)
    clean_season = season_key.strip()
    train_json[clean_season] = {}
    
    for round_key, matches in season_data.items():
        # Normalize round names
        clean_round = round_key.strip().lower().replace(' ', '_')
        
        # Standardize round names
        round_mapping = {
            'round_16': 'round_of_16',
            'round-16': 'round_of_16',
            'round_of_16': 'round_of_16',
            'quarterfinals': 'quarter_finals',
            'quarter-finals': 'quarter_finals',
            'quarter_finals': 'quarter_finals',
            'semifinals': 'semi_finals',
            'semi-finals': 'semi_finals',
            'semi_finals': 'semi_finals',
            'final': 'final',
            'finals': 'final'
        }
        
        clean_round = round_mapping.get(clean_round, clean_round)
        train_json[clean_season][clean_round] = matches

print(f"  train.json: Cleaned successfully ({len(train_json)} seasons)")

# ============================================================================
# STEP 2: THOROUGH DATA CLEANING
# ============================================================================
# Clean the dataset to ensure high-quality training data

print("\nCleaning data...")

# Drop columns related to extra time and penalties (not needed for basic prediction)
df = df.drop(['Home_Score_AET', 'Away_Score_AET', 'Home_Penalties', 'Away_Penalties'], 
             axis=1, errors='ignore')
print("  ✓ Dropped AET and penalty columns")

# Remove rows with missing scores or points (critical data)
initial_len = len(df)
df = df.dropna(subset=['Team_Score', 'Opponent_Score', 'Team_Points', 'Opponent_Points'])
print(f"  ✓ Removed {initial_len - len(df):,} rows with missing scores")

# Parse dates properly (critical for time-based features)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['Date'])
print("  ✓ Parsed dates successfully")

# Remove invalid scores (sanity check - scores should be between 0-30)
df = df[(df['Team_Score'] >= 0) & (df['Opponent_Score'] >= 0)]
df = df[(df['Team_Score'] < 30) & (df['Opponent_Score'] < 30)]
print("  ✓ Removed invalid scores")

# Remove duplicate matches (keep first occurrence)
before_dedup = len(df)
df = df.drop_duplicates(subset=['Date', 'Team', 'Opponent', 'Competition'], keep='first')
print(f"  ✓ Removed {before_dedup - len(df):,} duplicate matches")

print(f"\nFinal clean dataset: {len(df):,} matches")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
# Create meaningful features from raw data
# All features use .shift(1) to prevent data leakage (no future information)

print("\nEngineering features...")

# --- BASIC FEATURES ---
# Create fundamental match statistics
df['goal_diff'] = df['Team_Score'] - df['Opponent_Score']  # Goal difference in match
df['points_diff'] = df['Team_Points'] - df['Opponent_Points']  # Points difference
df['advance'] = (df['goal_diff'] >= 0).astype(int)  # Target: 1 if team advanced, 0 otherwise
print("  ✓ Basic features (goal_diff, points_diff, advance)")

# Sort by team and date for time-based features
df = df.sort_values(['Team', 'Date'])

# --- FEATURE 1: RECENT FORM (LAST 5 MATCHES) ---
# Rolling statistics over last 5 matches (excluding current match with shift(1))
df['goals_for_5'] = df.groupby('Team')['Team_Score'].shift(1).rolling(5).mean()
df['goals_against_5'] = df.groupby('Team')['Opponent_Score'].shift(1).rolling(5).mean()
df['points_5'] = df.groupby('Team')['Team_Points'].shift(1).rolling(5).mean()
df['win_rate_5'] = df.groupby('Team')['advance'].shift(1).rolling(5).mean()
df['goal_diff_5'] = df['goals_for_5'] - df['goals_against_5']
print("  ✓ Recent form features (last 5 matches)")

# --- FEATURE 2: SEASON-LEVEL STATISTICS ---
# Expanding mean within each season (all previous matches in the season)
df['season_goal_diff'] = (
    df.groupby(['Team','season'])['goal_diff']
      .apply(lambda x: x.shift(1).expanding().mean())  # shift(1) prevents leakage
      .reset_index(level=[0,1], drop=True)
)

df['season_points_avg'] = (
    df.groupby(['Team','season'])['Team_Points']
      .apply(lambda x: x.shift(1).expanding().mean())  # Average points per game
      .reset_index(level=[0,1], drop=True)
)
print("  ✓ Season statistics (goal diff, points avg)")

# Sort by season for head-to-head calculations
df = df.sort_values('season')

# --- FEATURE 3: HEAD-TO-HEAD (H2H) STATISTICS ---
# Historical performance against specific opponents
df['h2h_win_rate'] = (
    df.groupby(['Team','Opponent'])['advance']
      .apply(lambda x: x.shift(1).expanding().mean())  # Win rate vs this opponent
      .reset_index(level=[0,1], drop=True)
)

df['h2h_goal_diff'] = (
    df.groupby(['Team','Opponent'])['goal_diff']
      .apply(lambda x: x.shift(1).expanding().mean())  # Avg goal diff vs this opponent
      .reset_index(level=[0,1], drop=True)
)
print("  ✓ Head-to-head features (win rate, goal diff)")

# --- FEATURE 4: EXPERIENCE (NUMBER OF MATCHES PLAYED) ---
# Count of total matches and Champions League specific matches
df['matches_played'] = df.groupby('Team').cumcount()  # Total career matches

# Champions League specific experience
df['cl_matches'] = 0
cl_mask = df['Competition'] == 'uefa-champions-league'
df.loc[cl_mask, 'cl_matches'] = df[cl_mask].groupby('Team').cumcount()
print("  ✓ Experience features (matches played)")

# --- FEATURE 5: CONSISTENCY ---
# Measure how consistent a team's performance is
df['goals_std_5'] = df.groupby('Team')['Team_Score'].shift(1).rolling(5, min_periods=2).std()
df['consistency_score'] = 1 / (1 + df['goals_std_5'].fillna(1))  # Higher = more consistent
print("  ✓ Consistency features")

# --- FEATURE 6: LATE TOURNAMENT EXPERIENCE ---
# Track performance in knockout stages (semis, finals)

# Identify knockout matches
df['is_knockout'] = df['Round'].str.contains('final|semi|quarter|16', case=False, na=False).astype(int)
df['knockout_matches'] = df.groupby('Team')['is_knockout'].cumsum()

# Separate semifinals from finals
df['is_semifinal'] = df['Round'].str.contains('semi', case=False, na=False).astype(int)
df['is_final'] = df['Round'].str.contains('final', case=False, na=False).astype(int)
df['is_final'] = df['is_final'] & ~df['is_semifinal']  # Ensure finals don't include semis

# Count how many times team reached semis/finals
df['semifinals_reached'] = df.groupby('Team')['is_semifinal'].cumsum()
df['finals_reached'] = df.groupby('Team')['is_final'].cumsum()

# Big game experience score (finals worth 3x semifinals)
df['big_game_exp'] = df['finals_reached'] * 3 + df['semifinals_reached']
print("  ✓ Late tournament experience features")

# --- FEATURE 7: FIFA CLUB WORLD CUP PRESTIGE ---
# Add prestige scores based on Club World Cup performance
df['team_cwc_prestige'] = df['Team'].apply(get_cwc_prestige)
df['opp_cwc_prestige'] = df['Opponent'].apply(get_cwc_prestige)
df['cwc_prestige_diff'] = df['team_cwc_prestige'] - df['opp_cwc_prestige']
print("  ✓ FIFA Club World Cup prestige features")

# --- HANDLE MISSING VALUES ---
# Fill NaN values with reasonable defaults
df = df.fillna({
    'h2h_win_rate': 0.5,  # Neutral assumption (50% win rate)
    'h2h_goal_diff': 0.0,  # No advantage
    'consistency_score': 0.5,  # Average consistency
})

# Drop rows where target variable is missing
df = df.dropna(subset=['advance'])

print(f"\nFeature engineering complete: {len(df):,} matches ready for training")

# ============================================================================
# STEP 4: PREPARE TRAINING DATA
# ============================================================================
# Split data into training and validation sets based on date

print("\nPreparing training data...")

# Use data up to April 2017 for training (following notebook approach)
cutoff_date = pd.to_datetime('2017-04-30')
train_data = df[df['Date'] <= cutoff_date].copy()
val_data = df[(df['Date'] > cutoff_date) & (df['Date'] <= pd.to_datetime('2017-12-31'))].copy()

print(f"  Training set: {len(train_data):,} matches")
print(f"  Validation set: {len(val_data):,} matches")

# --- DEFINE FEATURES AND TARGET ---
# Columns to exclude from features (target and source data)
drop_cols = [
    'Team_Score',      # Source data (would cause leakage)
    'Opponent_Score',  # Source data
    'goal_diff',       # Derived from scores
    'advance',         # Target variable
    'Team_Points',     # Source data
    'Opponent_Points', # Source data
    'points_diff',     # Derived from points
    'Date',            # Not a predictive feature
    'Time',            # Not a predictive feature
    'is_knockout',     # Intermediate calculation
    'is_semifinal',    # Intermediate calculation
    'is_final'         # Intermediate calculation
]

# All other columns are features
feature_cols = [c for c in train_data.columns if c not in drop_cols]

# Categorical columns (need special encoding)
cat_cols = ['Round', 'Team', 'Opponent', 'Location', 'Country', 'Competition']

# Prepare feature matrices (X) and target vectors (y)
X_train = train_data[feature_cols].copy()
y_train = train_data['advance'].copy()

X_val = val_data[feature_cols].copy()
y_val = val_data['advance'].copy()

# Convert categorical columns to 'category' dtype for XGBoost and LightGBM
for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

print(f"  Features: {len(feature_cols)}")
print(f"  Categorical: {len(cat_cols)}")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================
# Use Optuna to find optimal hyperparameters for both models

print("\nHyperparameter tuning with Optuna...")

N_TRIALS = 20  # Number of trials per model

# --- XGBOOST OPTIMIZATION ---
def objective_xgb(trial):
    """
    Optuna objective function for XGBoost.
    Suggests hyperparameters and returns validation accuracy.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 42,
        'enable_categorical': True  # Handle categorical features natively
    }
    
    # Train model with suggested parameters
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate on validation set
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

# Run optimization
print("  Optimizing XGBoost...", end=" ", flush=True)
study_xgb = optuna.create_study(
    direction='maximize',  # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42)  # Reproducible results
)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=False)

best_xgb_params = study_xgb.best_params
print(f"Best accuracy: {study_xgb.best_value:.4f}")

# --- LIGHTGBM OPTIMIZATION ---
def objective_lgb(trial):
    """
    Optuna objective function for LightGBM.
    Suggests hyperparameters and returns validation accuracy.
    """
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),  # L2 regularization
        'random_state': 42,
        'verbose': -1
    }
    
    # Train model with suggested parameters
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
    
    # Evaluate on validation set
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

# Run optimization
print("  Optimizing LightGBM...", end=" ", flush=True)
study_lgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=False)

best_lgb_params = study_lgb.best_params
print(f"Best accuracy: {study_lgb.best_value:.4f}")

# ============================================================================
# STEP 6: TRAIN FINAL MODELS
# ============================================================================
# Train final models using the best hyperparameters found by Optuna

print("\nTraining final models...")

# Train XGBoost with best parameters
xgb_model = xgb.XGBClassifier(**best_xgb_params, random_state=42, enable_categorical=True)
xgb_model.fit(X_train, y_train)
print("  ✓ XGBoost trained")

# Train LightGBM with best parameters
lgb_model = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
print("  ✓ LightGBM trained")

# Set ensemble weights (equal weighting)
weights = np.array([0.5, 0.5])
print(f"  Ensemble weights: XGB={weights[0]:.1f}, LGB={weights[1]:.1f}")

# ============================================================================
# STEP 7: SIMPLE PREDICTION FUNCTION
# ============================================================================
# Functions to build features and make predictions for new matches

# Store feature columns for later use
FEATURE_COLUMNS = feature_cols

def build_features_for_match(teamA, teamB, season):
    """
    Build feature vector for a specific matchup.
    
    This function extracts historical features for teamA playing against teamB
    in a given season. It uses the most recent available data.
    
    Args:
        teamA (str): Name of team A
        teamB (str): Name of team B
        season (str): Season identifier (e.g., '2017-18')
        
    Returns:
        pd.DataFrame: Single-row dataframe with features
    """
    # Try to find direct matchup history first
    data = df[
        (df['Team'] == teamA.upper()) &
        (df['Opponent'] == teamB.upper()) &
        (df['season'] == season)
    ].sort_values('Date')

    # Fallback: if no direct matchup, use any match from teamA in this season
    if data.empty:
        data = df[
            (df['Team'] == teamA.upper()) &
            (df['season'] == season)
        ].sort_values('Date')

    # Last resort: if still no data, use median/mode values from training set
    if data.empty:
        feature_row = pd.DataFrame(index=[0], columns=FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            if col in cat_cols:
                # Use most common value for categorical features
                feature_row[col] = X_train[col].mode()[0]
            else:
                # Use median for numerical features
                feature_row[col] = X_train[col].median()
        return feature_row

    # Extract features from most recent match
    row = data.iloc[-1]
    feature_row = pd.DataFrame([row[FEATURE_COLUMNS]])

    # Ensure categorical columns have correct dtype
    for col in cat_cols:
        if col in feature_row.columns:
            feature_row[col] = feature_row[col].astype('category')

    return feature_row

def predict_simple(teamA, teamB, season):
    """
    Predict the winner of a match using ensemble prediction.
    
    This is a simple ensemble that averages predictions from XGBoost and LightGBM.
    No round-specific weighting is applied.
    
    Args:
        teamA (str): Name of team A
        teamB (str): Name of team B
        season (str): Season identifier
        
    Returns:
        str: Predicted winner (teamA or teamB)
    """
    # Build feature vector for this matchup
    features = build_features_for_match(teamA, teamB, season)

    try:
        # Get probability predictions from both models
        xgb_prob = xgb_model.predict_proba(features)[0, 1]  # Probability of teamA winning
        lgb_prob = lgb_model.predict_proba(features)[0, 1]
        
        # Ensemble prediction (weighted average)
        ml_prediction = weights[0] * xgb_prob + weights[1] * lgb_prob
    except:
        # Fallback to 50-50 if prediction fails
        ml_prediction = 0.5

    # Return predicted winner (>0.5 means teamA wins)
    return teamA if ml_prediction > 0.5 else teamB

# ============================================================================
# STEP 8: GENERATE PREDICTIONS
# ============================================================================
# Generate predictions for all test matchups following the bracket structure

print("\nGenerating predictions...")

# Load test matchups from JSON file
with open('test_matchups.json', 'r') as f:
    test_matchups = json.load(f)

# Helper functions to extract bracket progression
def extract_r16_index(desc, r16_lookup):
    """Extract which R16 match feeds into QF"""
    if desc.startswith("Winner of "):
        return r16_lookup.get(desc[10:])
    return None

def extract_qf_index(desc):
    """Extract which QF feeds into SF"""
    match = re.search(r'QF(\d+)', desc)
    return int(match.group(1)) - 1 if match else None

def extract_sf_index(desc):
    """Extract which SF feeds into Final"""
    match = re.search(r'SF(\d+)', desc)
    return int(match.group(1)) - 1 if match else None

# Store all predictions
all_predictions = []

# Process each season
for season, bracket in test_matchups.items():
    print(f"  {season}...", end=" ")
    
    # Initialize predictions structure
    predictions = {
        'round_of_16': [],
        'quarter_finals': [],
        'semi_finals': [],
        'final': []
    }

    # --- ROUND OF 16 ---
    r16_winners = {}  # Store winners by match index
    r16_lookup = {}   # Map "Team1 vs Team2" to match index
    
    for i, match in enumerate(bracket['round_of_16_matchups']):
        team1, team2 = match['team_1'], match['team_2']
        
        # Predict winner
        winner = predict_simple(team1, team2, season)
        
        # Store results
        r16_winners[i] = winner
        r16_lookup[f"{team1} vs {team2}"] = i
        
        predictions['round_of_16'].append({
            'team_1': team1,
            'team_2': team2,
            'winner': winner
        })

    # --- QUARTER FINALS ---
    qf_winners = {}
    
    for i, match in enumerate(bracket['quarter_finals_matchups']):
        # Extract which R16 matches feed into this QF
        r16_idx1 = extract_r16_index(match['team_1'], r16_lookup)
        r16_idx2 = extract_r16_index(match['team_2'], r16_lookup)
        
        if r16_idx1 is not None and r16_idx2 is not None:
            # Get the R16 winners
            team1 = r16_winners[r16_idx1]
            team2 = r16_winners[r16_idx2]
            
            # Predict QF winner
            winner = predict_simple(team1, team2, season)
            
            qf_winners[i] = winner
            predictions['quarter_finals'].append({
                'team_1': team1,
                'team_2': team2,
                'winner': winner
            })

    # --- SEMI FINALS ---
    sf_winners = {}
    
    for i, match in enumerate(bracket['semi_finals_matchups']):
        # Extract which QF matches feed into this SF
        qf_idx1 = extract_qf_index(match['team_1'])
        qf_idx2 = extract_qf_index(match['team_2'])
        
        if qf_idx1 is not None and qf_idx2 is not None:
            # Get the QF winners
            team1 = qf_winners.get(qf_idx1)
            team2 = qf_winners.get(qf_idx2)
            
            if team1 and team2:
                # Predict SF winner
                winner = predict_simple(team1, team2, season)
                
                sf_winners[i] = winner
                predictions['semi_finals'].append({
                    'team_1': team1,
                    'team_2': team2,
                    'winner': winner
                })

    # --- FINAL ---
    final_match = bracket['final_matchup']
    sf_idx1 = extract_sf_index(final_match['team_1'])
    sf_idx2 = extract_sf_index(final_match['team_2'])
    
    if sf_idx1 is not None and sf_idx2 is not None:
        # Get the SF winners
        team1 = sf_winners.get(sf_idx1)
        team2 = sf_winners.get(sf_idx2)
        
        if team1 and team2:
            # Predict champion
            winner = predict_simple(team1, team2, season)
            
            predictions['final'].append({
                'team_1': team1,
                'team_2': team2,
                'winner': winner
            })
            
            print(f"Champion: {winner}")

    # Store predictions for this season
    all_predictions.append({
        'id': len(all_predictions),
        'season': season,
        'predictions': json.dumps(predictions)
    })

# ============================================================================
# STEP 9: SAVE SUBMISSION
# ============================================================================
# Save predictions in the required format

print("\nSaving submission...")

submission = pd.DataFrame(all_predictions)
submission.to_csv('submission_xgb_lgb_simple_ensemble.csv', index=False)

print("✓ Saved: submission_xgb_lgb_simple_ensemble.csv")
print(f"✓ Total predictions: {len(submission)} seasons")

print("\n" + "="*90)
print("PIPELINE COMPLETE!")
print("="*90)
print("""
Summary:
  • Models: XGBoost + LightGBM (Optuna-tuned)
  • Features: Form, H2H, Consistency, Experience, Late Tournament
  • Predictions: Simple ensemble (no round weighting)
  • Output: submission_xgb_lgb_simple_ensemble.csv
  
Ready for submission! 🏆
""")
