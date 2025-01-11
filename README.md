# shmowt
Structural Health Monitoring for Jacket-Type Offshore Wind Turbines.

This project aims to replicate the methodology and results of the paper[1] [accessible here](https://www.mdpi.com/1424-8220/20/7/1835).

It also proposes a few improvements described in the companion report (unpublished).

[1] Vidal Y, Aquino G, Pozo F, Gutiérrez-Arias JEM. Structural Health Monitoring for Jacket-Type Offshore Wind Turbines: Experimental Proof of Concept. Sensors (Basel). 2020 Mar 26;20(7):1835. doi: 10.3390/s20071835. PMID: 32224918; PMCID: PMC7180893.

## Installation and usage
Dependencies are managed by [poetry](https://python-poetry.org/docs/#installation), so you must install that first.

Clone the repository:

```bash
git clone git@github.com:compi-migui/shmowt.git
cd shmowt
```
Install dependencies:

```bash
poetry install
```
Run shmowt:

```bash
SHMOWT_CONFIG=~/shmowt/contrib/config.ini poetry run 
```
