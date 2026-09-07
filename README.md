# Risk Pulse Lab v0.1

Web-based operational-risk training and decision-support prototype.

## What it does
- simple objective/exposure-first interface
- public-data frequency anchors where defensible
- Gamma-Poisson Bayesian updating with optional company evidence
- posterior predictive Monte Carlo
- Objective-at-Risk / materiality probability
- public ML challenger backtest with a hard promotion rule
- source registry that can refresh CFPB and VCDB and check FCA availability
- QR sharing in the browser

## Model governance
The bundled pilot backtest uses 50 FCA product series. A machine-learning challenger is not promoted when it fails to beat the persistence benchmark. Complaint data are treated as a frequency proxy, not as operational-loss amounts.

The process/cyber absolute frequency priors are explicitly labelled scenario priors because public incident databases do not provide unbiased company exposure denominators. User-supplied internal event and severity evidence should replace them for client use.

## Netlify deployment
This repository is structured so Netlify can publish directly from the repository root. Use `main` as the production branch and `.` as the publish directory.

## Python research engine
The original Python research/model files are kept under `research/` for model development, public-data ingestion, and backtesting. The production MVP runs client-side from `index.html`.
