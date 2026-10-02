#!/usr/bin/env bash
# Reproduce every table and figure in the paper. Usage: bash run_all.sh [workers]
set -e
W=${1:-8}
cd "$(dirname "$0")/code"
python verify_propositions.py | tee ../output/verify_propositions.log   # Propositions 1-2, Lemma 1 (~2 min)
python run_simulations.py --workers "$W"                                 # Section 3 (hours; resumable)
python run_realdata.py --workers "$W"                                    # Section 4 (requires ../data/working_data_dedup.pkl)
python make_outputs.py                                                   # tables, figures, numbers_for_text.md
