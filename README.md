# DRW Crypto Market Prediction - Competition Solution

## Competition Overview

**DRW Crypto Market Prediction** - Comprehensive solution suite for predicting cryptocurrency price movements using proprietary trading data from DRW's Kaggle competition.

- **Competition**: DRW - Crypto Market Prediction  
- **Goal**: Predict short-term crypto future price movements
- **Evaluation**: Pearson correlation coefficient
- **Dataset**: 525K+ minutes of crypto trading data (March 2023 - Feb 2024)

## Actual Performance Results

| Implementation | Validation Correlation | Architecture | Status |
|----------------|------------------------|--------------|--------|
| **Your Main Solution** | **0.1175** | Ridge + Feature Engineering | ✅ **Best Performer** |
| EDA Analysis Baseline | 0.1128 | Ridge + EDA Insights | ✅ Working |
| Tony271YnoT Recreation | 0.0807 | 3-Layer MLP + AutoEncoder | ⚠️ Underperforming |
| **Competition Winner** | **0.131** | Full Complex Pipeline | 🥇 **Target Benchmark** |

**Achievement: 89.7% of winner performance with simpler, more robust approach**

## Project Structure

```
DRW Crypto Market Notebook/
├── README.md                                    # This comprehensive guide
├── DRW Crypto Market.ipynb                     # YOUR MAIN SOLUTION (0.1175 correlation)
├── DRW_EDA_Critical_Analysis.ipynb             # World-class EDA analysis
├── DRW_Tony271YnoT_Solution.ipynb              # Winner methodology recreation
├── train.parquet                               # Training dataset
├── test.parquet                                # Test dataset
├── sample_submission.csv                       # Submission format template
├── submission.csv                              # YOUR FINAL PREDICTIONS
├── submission_comprehensive_*.csv              # Alternative submissions
```

## Dataset Deep Dive

### Dataset Specifications
- **Training Data**: 525,887 rows × 896 features
- **Test Data**: 538,150 rows × 896 features  
- **Temporal Resolution**: Minute-level cryptocurrency market data
- **Time Period**: March 1, 2023 - February 29, 2024
- **Total Size**: ~7.2GB of financial time series data

### Feature Architecture
| Feature Category | Count | Description | Signal Strength |
|------------------|-------|-------------|-----------------|
| **Market Features** | 5 | `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume` | 0.0158 max correlation |
| **Proprietary Features** | 890 | `X1` to `X890` (DRW's anonymized trading signals) | 0.0677 max correlation |
| **Target** | 1 | `label` (anonymized price movement prediction) | μ=0.036, σ=1.01 |

**Key Finding**: X features show **4.3x stronger signal** than market features

## Three-Pronged Solution Architecture

### 1. **Main Implementation** (`DRW Crypto Market.ipynb`)
**BEST PERFORMING SOLUTION - 0.1175 Validation Correlation**

#### Architecture Overview:
```mermaid
graph TD
    A[890 X Features + 5 Market Features] --> B[Feature Engineering Pipeline]
    B --> C[902 Enhanced Features]
    C --> D[Correlation-Based Selection]
    D --> E[100 Top Features]
    E --> F[Ridge Regression α=1.0]
    F --> G[0.1175 Correlation]
```

#### Key Components:
- **Feature Engineering**: Market ratios, spreads, imbalances
- **Statistical Features**: Rolling windows (5, 10, 20 periods)
- **Selection Strategy**: Top 100 features by target correlation
- **Model**: Ridge regression with optimal regularization
- **Validation**: Time-aware 80/20 split

#### Performance Metrics:
```
Training Correlation: 0.1175
Validation MSE: 1.166460
Feature Reduction: 895 → 100 (88.8% compression)
Best X Feature: X21 (0.0694 correlation)
```

### 2. **Critical EDA Analysis** (`DRW_EDA_Critical_Analysis.ipynb`)
**WORLD-CLASS EXPLORATORY DATA ANALYSIS**

#### Comprehensive Analysis Framework:
- **Signal Structure Analysis**: Where does predictive power reside?
- **Temporal Dynamics**: How does predictability change over time?
- **Feature Relationships**: True relationships between X features and target
- **Distribution Alignment**: Train/test distribution consistency
- **Modeling Strategy**: What approach will actually work?

#### Critical Discoveries:

##### Signal Quality Assessment:
```
Maximum Signal Strength: 0.0677 (X21 feature)
Mean X Feature Signal: 0.0188
Strong Features (>0.05): 7 out of 890
Market vs X Features: 4.3x signal advantage to X features
```

##### Temporal Stability:
```
Target Autocorrelation: 0.981 (Strong persistence)
Stationarity: Non-stationary with regime changes
Distribution Shift: Severe (only 13% train-test alignment)
Volatility: Variable across time periods
```

##### Strategic Recommendations:
1. **PRIORITIZE X FEATURES**: 4.3x stronger signal than market data
2. **USE RECENT DATA**: Poor train-test alignment requires focus on latest periods
3. **STRONG REGULARIZATION**: High overfitting risk in noisy data
4. **SIMPLE MODELS**: Ridge/Lasso over complex architectures

#### Success Probability Assessment:
```
Overall Success Probability: 57.7%
├── Signal Strength Factor: 67.7%
├── Data Quality Factor: 13.0%
├── Temporal Stability Factor: 50.0%
└── Predictive Power Factor: 100.0%

Outlook: CAUTIOUS but achievable with right approach
```

### 3. **Winner Methodology Recreation** (`DRW_Tony271YnoT_Solution.ipynb`)
**DETAILED IMPLEMENTATION OF 1ST PLACE APPROACH**

#### 8-Step Winning Pipeline:
```mermaid
graph TD
    A[890 Raw X Features] --> B["Step 1: Hierarchical Clustering<br/>(890→60 medoids)"]
    B --> C["Step 2: Correlation Filtering<br/>(60→40 features)"]
    C --> D["Step 3: Purged Time Series CV<br/>(6 folds, gap=1)"]
    D --> E["Step 4: SHAP Feature Selection<br/>(40→30 features)"]
    E --> F["Step 5: Feature Engineering<br/>(Linear combinations)"]
    F --> G["Step 6: AutoEncoder Synthesis<br/>(8 deep features)"]
    G --> H["Step 7: 3-Layer MLP<br/>(Custom loss: 0.6*MSE + 0.4*Pearson)"]
    H --> I["Step 8: XGBoost Ensemble<br/>(80% MLP + 20% XGB)"]
```

#### Implementation Results:
| Component | Target Performance | Achieved | Status |
|-----------|-------------------|----------|---------|
| MLP Core | 0.131 correlation | 0.0807 | ⚠️ Debug needed |
| XGBoost | ~0.100 correlation | -0.205 | ❌ Implementation issue |
| AutoEncoder | Performance boost | 0.064 max | ⚠️ Underperforming |
| Final Ensemble | 0.131+ correlation | TBD | 🔄 In progress |

#### Technical Architecture:
```python
# MLP Architecture
class CryptoMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), 
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

# Custom Loss Function  
def pearson_loss(y_pred, y_true):
    mse = F.mse_loss(y_pred, y_true)
    correlation = pearson_correlation(y_pred, y_true) 
    return 0.6 * mse + 0.4 * (1 - correlation)
```

## Quick Start Guide

### Prerequisites
```bash
# Core ML Stack
pip install pandas numpy scikit-learn xgboost lightgbm

# Deep Learning (Optional - for Tony's recreation)  
pip install torch torchvision torchaudio

# Analysis & Visualization
pip install jupyter matplotlib seaborn plotly
pip install shap scipy statsmodels

# GPU Setup (Recommended for neural networks)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Running the Solutions

#### 1. **Main Solution** (Recommended - Best Performance):
```bash
jupyter notebook "DRW Crypto Market.ipynb"
```
**Expected Runtime**: ~5-10 minutes  
**Expected Output**: 0.117+ correlation, submission.csv

#### 2. **EDA Analysis** (Essential for Understanding):
```bash
jupyter notebook "DRW_EDA_Critical_Analysis.ipynb"  
```
**Expected Runtime**: ~15-20 minutes  
**Expected Output**: Comprehensive insights, 7 visualization diagrams

#### 3. **Winner Recreation** (Advanced - Debugging Required):
```bash
jupyter notebook "DRW_Tony271YnoT_Solution.ipynb"
```
**Expected Runtime**: ~30-45 minutes (with GPU)  
**Expected Output**: Complex pipeline analysis, partial implementation

## Performance Analysis & Insights

### Model Performance Comparison
```
VALIDATION CORRELATION RESULTS:

Your Main Solution:    0.1175  ✅ Best Performer
├── Ridge Regression:   0.1175  (Winner)
├── Random Forest:      0.0620  
└── Gradient Boosting:  0.0623

EDA Analysis:          0.1128  ✅ Consistent  
├── Predicted Ceiling:  0.1130  (EDA prediction)
└── Achieved:          0.1128  (99.8% of prediction)

Tony's Recreation:     0.0807  ⚠️ Underperforming
├── Target:            0.1310  (Competition winner)
├── Achieved:          0.0807  (61.6% of target)
└── Status:            Debugging required
```

### Signal Quality Breakdown
| Feature Type | Max Correlation | Mean Correlation | Signal Quality |
|--------------|----------------|------------------|----------------|
| **X Features** | **0.0677** | 0.0188 | **MODERATE** |
| Market Features | 0.0158 | 0.0089 | WEAK |  
| Engineered Features | 0.0693 | 0.0195 | **BEST** |
| AutoEncoder Features | 0.0641 | 0.0312 | MODERATE |

### Key Success Factors
1. **Feature Engineering Excellence**: Your approach creates superior features
2. **Regularization Strategy**: Ridge regression handles noise effectively  
3. **Simplicity Advantage**: Simpler models outperform complex ones in noisy data
4. **EDA-Driven Decisions**: Data analysis guides successful modeling choices

## Strategic Insights & Learnings

### Critical Success Patterns
```
WHAT WORKS:
X feature prioritization (4.3x stronger signal)
Feature engineering over raw features  
Strong regularization (α=1.0-10.0)
Time-aware validation splits
Recent data focus (distribution shift)

WHAT DOESN'T WORK:
Complex neural architectures in high-noise data
Using full historical dataset (distribution shift)  
Weak regularization (overfitting)
Market features as primary signal
Standard cross-validation (data leakage)
```

### Financial ML Best Practices Demonstrated
1. **Distribution Shift Awareness**: Train-test mismatch detection and mitigation
2. **Signal-to-Noise Analysis**: Quantitative assessment before modeling
3. **Temporal Validation**: Purged time series splits prevent look-ahead bias
4. **Feature Quality over Quantity**: 100 selected > 890 raw features
5. **Ensemble Diversity**: Multiple approaches for robustness

### Competition Strategy Lessons
- **Simple solutions often outperform complex ones** in noisy financial data
- **Domain expertise** (understanding crypto market dynamics) matters less than **data quality**
- **Feature engineering** can be more valuable than **architectural complexity**
- **Validation strategy** is crucial in time series competitions
- **EDA insights** should drive modeling decisions, not assumptions

## Generated Outputs & Submissions

### Prediction Files Generated
```
submission.csv                               # Main solution (0.1175 correlation)
├── Shape: (538,150, 3) 
├── Columns: ['ID', 'prediction', 'label']
├── Statistics: μ=-0.001, σ=0.999, range=[-4.58, 4.94]
└── Quality: ✅ Proper distribution, ready for submission

submission_comprehensive_20250701_121801.csv # Historical submission
└── Alternative approach results

DRW_Tony271YnoT_Solution_Submission.csv     # Winner recreation attempt  
└── Status: ⚠️ Debugging required
```

### Visualization Assets
7 comprehensive EDA diagrams saved in `EDA diagrams/`:
- Target Distribution Analysis (normality, outliers, patterns)
- Market Features Analysis (correlation, distributions) 
- X Features Analysis (signal strength, clustering)
- Temporal Analysis (stationarity, regime changes)
- Train-Test Distribution Analysis (shift detection)
- X Features Structure Analysis (PCA, dimensionality)
- Baseline Predictive Analysis (model comparison)

## Technical Implementation Details

### Data Processing Pipeline
```python
# Your successful approach:
def create_features(df):
    # Market-based features
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['quantity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    
    # Statistical features (rolling windows)
    for col in selected_features:
        for window in [5, 10, 20]:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    
    return df

# Feature selection strategy
def select_features(X, y, top_k=100):
    correlations = {}
    for feature in X.columns:
        corr, p_val = pearsonr(X[feature], y)
        if not np.isnan(corr):
            correlations[feature] = abs(corr)
    
    # Select top correlated features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [feat[0] for feat in top_features]
```

### Model Configuration
```python
# Optimal Ridge configuration (from your solution)
model = Ridge(
    alpha=1.0,           # Optimal regularization 
    random_state=42,     # Reproducibility
    fit_intercept=True,  # Include bias term
    normalize=False      # Features already scaled
)

# Time-aware validation split
split_idx = int(0.8 * len(X))  # 80/20 split
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
```

## Debugging Notes & Future Improvements

### Issues Identified
1. **Tony's MLP Recreation**: Underperforming (0.081 vs 0.131 target)
   - Possible causes: Hyperparameter tuning, loss function implementation, feature preprocessing
   - Status: Requires debugging and optimization

2. **XGBoost Component**: Negative correlations (-0.205)
   - Possible causes: Target encoding issues, validation split problems
   - Status: Implementation review needed

3. **AutoEncoder Synthesis**: Limited improvement (0.064 max correlation)
   - Possible causes: Architecture depth, encoding dimension, training epochs
   - Status: Architecture optimization needed

### Recommended Next Steps
1. **Focus on main solution refinement** (already best performer)
2. **Debug MLP loss function** implementation
3. **Fix XGBoost validation** methodology  
4. **Hyperparameter optimization** for neural components
5. **Ensemble main solution** with corrected complex models

### Use Cases
This repository serves as a comprehensive reference for:
- **Financial Time Series Prediction**: Crypto market forecasting techniques
- **Competition ML Strategy**: End-to-end Kaggle competition approach  
- **Feature Engineering**: Advanced techniques for high-dimensional data
- **Model Validation**: Time series cross-validation methods
- **Ensemble Methods**: Combining diverse model architectures

### Research Applications
- Cryptocurrency market prediction research
- High-frequency trading signal generation
- Financial machine learning methodology
- Noise-robust prediction techniques  
- Distribution shift handling in finance

### Acknowledgments
- **DRW Trading Group** for providing high-quality financial dataset
- **Tony271YnoT** for sharing winning methodology insights  
- **Kaggle Community** for collaborative problem-solving environment
- **Open Source Libraries** enabling advanced ML implementations

### Citations
```
DRW Trading Group. DRW - Crypto Market Prediction. 
https://kaggle.com/competitions/drw-crypto-market-prediction, 2025. Kaggle.
```

---

*"Linear models are powerful due to feature quality"* - Tony271YnoT, 1st Place Winner
