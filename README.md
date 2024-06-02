# shmowt
Structural Health Monitoring for Jacket-Type Offshore Wind Turbines.

This project aims to reproduce the methodology and results of the paper[1] [accessible here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180893/).

It may eventually go beyond that by adding new features, to be determined.

[1] Vidal Y, Aquino G, Pozo F, Gutiérrez-Arias JEM. Structural Health Monitoring for Jacket-Type Offshore Wind Turbines: Experimental Proof of Concept. Sensors (Basel). 2020 Mar 26;20(7):1835. doi: 10.3390/s20071835. PMID: 32224918; PMCID: PMC7180893.

## Usage

```bash
git clone git@github.com:compi-migui/shmowt.git
cd shmowt
poetry install
SHMOWT_CONFIG=~/shmowt/contrib/config.ini poetry run 
```