### Portfolio Optimization: Synthesizing Ensemble Machine Learning, Random Matrix Theory, Tyler's M-Estimation, and Dynamic Factor Constraints for Hedge Fund Indices

#### Table of Contents
- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Implementation Details](#implementation-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

This research presents an advanced portfolio optimization framework that significantly outperforms traditional Mean-Variance Optimization (MVO) by integrating cutting-edge quantitative finance techniques. The methodology combines Random Matrix Theory (RMT) for noise filtering, Tyler's M-Estimator for robust covariance estimation, ensemble machine learning for expected returns prediction, and dynamic factor-based constraints for systematic risk management.

The framework demonstrates substantial improvements over static MVO benchmarks across multiple risk-adjusted performance metrics during the study period from January 2005 to May 2023.

## Key Contributions

### Technical Innovations
- **Self-implemented Random Matrix Theory noise filtering** using Marchenko-Pastur eigenvalue bounds
- **Custom Tyler's M-Estimator implementation** for outlier-robust covariance matrix estimation
- **Ensemble machine learning architecture** combining Huber Regression, AdaBoost, and ElasticNet models
- **Dynamic factor-based constraints** optimized through rolling Sharpe ratio maximization
- **Integrated optimization pipeline** with quarterly rebalancing over 46 overlapping 7-year windows

### Performance Achievements
- Superior risk-adjusted returns compared to traditional MVO approaches
- Enhanced portfolio diversification through systematic factor exposure management
- Robust performance across varying market conditions and volatility regimes
- Predictive model accuracy achieving 37-69% R² scores with 77-85% directional accuracy

## Methodology

### Data Sources and Scope
- **Assets**: Four HFRI hedge fund strategy indices (HFRI4FWC, HFRI4ELS, HFRI4EHV, HFRI4ED)
- **Factors**: Fama-French factors (Market, SMB, HML, MOM) and risk-free rate
- **Macro Indicators**: VIX, Yield Curve, Credit Spread and PMI
- **Period**: January 2005 - May 2023 (monthly frequency)
- **Framework**: Rolling 7-year training windows with quarterly portfolio rebalancing

### Core Components

#### 1. Factor Beta Estimation and Dynamic Constraints
The framework calculates factor loadings using rolling regression:

```
β_market[i] = Cov(Return_asset[i], Mkt-RF) / Var(Mkt-RF)
β_SMB[i] = Cov(Return_asset[i], SMB) / Var(SMB)
β_HML[i] = Cov(Return_asset[i], HML) / Var(HML)  
β_Mom[i] = Cov(Return_asset[i], Mom) / Var(Mom)
```

Dynamic constraints are optimized through four risk tolerance methodologies:
- Conservative Range (Interquartile): [Q₁, Q₃]
- Moderate Range: [P₁₀, P₉₀] 
- Aggressive Range: [μ - 1.5σ, μ + 1.5σ]
- Historical Range: [min(β) - 0.05, max(β) + 0.05]

#### 2. Robust Covariance Estimation Pipeline

**Tyler's M-Estimator Implementation:**
```
S(k+1) = (N/T) × Σ[x(t) × x(t)' / (x(t)' × S(k)⁻¹ × x(t))]
```

**RMT Noise Filtering:**
- Eigenvalue threshold: λ(max) = (1 + √(T/N))²
- Signal preservation for eigenvalues > λ(max)
- Noise eigenvalue averaging for λ ≤ λ(max)

#### 3. Ensemble Machine Learning for Expected Returns

**Feature Engineering:**
- Macroeconomic indicators: VIX, Yield Curve, Credit Spreads, PMI
- Factor returns: Market Factor, Size Factor

**Model Architecture:**
```
Expected Return = Huber Regression (40%) + AdaBoost (40%) + ElasticNet (20%)
```

**Mathematical Relationship:**
```
HF_Return(i,t) = f(VIX(t), YieldCurve(t), CreditSpread(t), PMI(t), MktRF(t), RF(t)) + ε(t)
```

#### 4. Portfolio Optimization Framework

**Objective Function:**
```
Utility = w'μ - (λ/2) × w'Σw
```

**Factor-Based Constraints:**
```
0.6 ≤ Σwᵢβᵢᵐᵏᵗ ≤ 1.1    (Market beta bounds)
-0.2 ≤ Σwᵢβᵢˢᵐᵇ ≤ 0.3    (Size beta bounds)  
-0.4 ≤ Σwᵢβᵢʰᵐˡ ≤ 0.2    (Value beta bounds)
-0.1 ≤ Σwᵢβᵢᵐᵒᵐ ≤ 0.2    (Momentum beta bounds)
```

## Project Structure

```
portfolio-optimization/
├── notebooks/
│   ├── beta_calculation.ipynb          # Factor beta computation
│   ├── beta_range.ipynb               # Dynamic constraint optimization  
│   ├── expected_returns.ipynb         # Ensemble ML return prediction
│   ├── rmt_filtered_covariance.ipynb  # RMT noise filtering implementation
│   ├── port_optimization.ipynb        # Main optimization engine
│   ├── rolling_backtest.ipynb         # Backtesting framework
│   ├── static_mvo.ipynb              # Benchmark MVO implementation
│   └── comparison.ipynb               # Performance comparison analysis
├── data/
│   ├── hedge_funds_returns_data.xlsx  # HFRI strategy index returns
│   ├── factors_returns_data.xlsx      # Fama-French factor data
│   └── market_signals_indexes.xlsx    # Macroeconomic predictors
├── all_output_results/
│   ├── All_Hedge_Fund_Betas/         # Factor loading calculations
│   ├── ML_validation_outputs/         # Model validation metrics
│   ├── ML_Ensemble_Models/           # Trained ensemble models
│   └── performance_metrics/           # Portfolio performance data
├── requirements.txt
├── METHODOLOGY.pdf
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0  
scipy>=1.7.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0
tqdm>=4.62.0
```

## Usage

### Execution Sequence

The analysis follows a structured pipeline across eight main notebooks:

1. **Factor Beta Calculation**
```bash
jupyter notebook notebooks/beta_calculation.ipynb
```
Computes rolling factor loadings for each hedge fund strategy using Fama-French four-factor model.

2. **Dynamic Constraint Optimization**
```bash  
jupyter notebook notebooks/beta_range.ipynb
```
Determines optimal factor exposure bounds through Sharpe ratio maximization across different risk tolerance frameworks.

3. **Expected Returns Generation**
```bash
jupyter notebook notebooks/expected_returns.ipynb  
```
Implements ensemble machine learning model for forward-looking return predictions using macroeconomic features.

4. **Robust Covariance Estimation**
```bash
jupyter notebook notebooks/rmt_filtered_covariance.ipynb
```
Applies Tyler's M-Estimator and RMT noise filtering to generate clean covariance matrices.

5. **Portfolio Optimization**
```bash
jupyter notebook notebooks/port_optimization.ipynb
```
Executes mean-variance optimization with factor-based constraints and filtered inputs.

6. **Rolling Backtesting**
```bash
jupyter notebook notebooks/rolling_backtest.ipynb
```
Performs out-of-sample testing with quarterly rebalancing over 46 overlapping periods.

7. **Benchmark Implementation**
```bash
jupyter notebook notebooks/static_mvo.ipynb
```
Implements traditional static MVO for performance comparison.

8. **Performance Analysis**
```bash
jupyter notebook notebooks/comparison.ipynb
```
Conducts comprehensive performance attribution and statistical comparison.

## Results

### Portfolio Performance Comparison

| Metric | Our Model (λ=0.05) | Our Model (λ=10) | Static MVO Benchmark |
|--------|---------------------|-------------------|---------------------|
| Annualized Return | 8.67% | 7.29% | ~6.42% |
| Volatility | 7.43% | 8.67% | ~9.39% |
| Sharpe Ratio | 1.15 | 0.84 | ~0.68 |
| Maximum Drawdown | 10.48% | 16.17% | ~15.40% |

### Machine Learning Model Validation

| Strategy Index | R² Score | Hit Rate | Breakdown Rate | Model Performance |
|---------------|----------|----------|----------------|------------------|
| HFRI4EHV | 68.7% | 85% | 0% | Excellent |
| HFRI4ELS | 62.9% | 82% | 2.1% | Strong |  
| HFRI4FWC | 49.4% | 77% | 4.3% | Good |
| HFRI4EMN | 37.4% | 79% | 1.8% | Moderate |

### Key Performance Insights

**Risk Management Enhancement:**
- Reduced maximum drawdown through robust covariance estimation
- Enhanced diversification via systematic factor exposure control
- Superior risk-adjusted returns across different risk aversion parameters

**Model Reliability:**  
- Consistent outperformance across 46 quarterly rebalancing periods
- Low breakdown rates indicating robust model stability
- High directional accuracy supporting practical implementation

**Innovation Impact:**
- First documented application combining RMT filtering with Tyler's M-Estimator for hedge fund portfolio optimization
- Demonstrates practical value of advanced statistical techniques in institutional portfolio management

## Implementation Details

### Technical Architecture

The implementation leverages entirely custom algorithms for the core statistical methods:

**Tyler's M-Estimator:**
- Iterative convergence algorithm with numerical stability controls
- Automatic regularization for singular matrix handling  
- Convergence monitoring with maximum iteration limits

**Random Matrix Theory Filtering:**
- Marchenko-Pastur eigenvalue threshold calculation
- Signal-noise eigenvalue separation with theoretical bounds
- Positive definite matrix reconstruction guarantees

**Ensemble Machine Learning:**
- Cross-validated hyperparameter optimization
- Temporal stability validation across rolling windows
- Robust performance metrics with breakdown rate monitoring

### Computational Considerations

- **Memory Efficiency:** Rolling window processing to manage computational complexity
- **Numerical Stability:** Regularization techniques for matrix operations
- **Scalability:** Modular design supporting extended asset universes
- **Reproducibility:** Fixed random seeds and deterministic algorithms

## Contributing

Contributions are welcome through the following process:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement-name`)
3. Implement changes with appropriate documentation
4. Add relevant tests and validation
5. Submit a pull request with detailed description

Please ensure all contributions maintain the academic rigor and documentation standards of the existing codebase.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete terms and conditions.

## References

1. Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. *The Annals of Statistics*, 15(1), 234-251.

2. Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Matematicheskii Sbornik*, 114(4), 507-536.

3. Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *The Annals of Statistics*, 48(5), 3043-3065.

4. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

5. Izmailov, A., & Shay, B. (2013). Dramatically improved portfolio optimization results with noise-filtered covariance. *Market Memory Trading Research Papers*.

---

**Contact:** For research inquiries or collaboration opportunities, please open an issue in this repository.

**Citation:** If you use this work in academic research, please cite the accompanying methodology paper and this implementation.
