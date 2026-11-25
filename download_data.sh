#!/usr/bin/env bash
# Downloads the public datasets into data/external/
set -e
cd "$(dirname "$0")/.."
mkdir -p data/external
cd data/external

clone() {
  if [ -d "$2" ]; then echo "already have $2"; else git clone --depth 1 "$1" "$2"; fi
}

clone https://github.com/l3cube-pune/code-mixed-nlp.git code-mixed-nlp            # L3Cube-HingLID (language ID)
clone https://github.com/SilentFlame/Named-Entity-Recognition.git Named-Entity-Recognition  # Singh et al. 2018 NER
clone https://github.com/20118/ACSA-Mix.git ACSA-Mix                               # aspect category + sentiment

echo
echo "Done. Optional, manual downloads:"
echo "  SentiMix (SemEval-2020 Task 9): https://zenodo.org/record/3974927  -> unzip into data/external/sentimix/"
echo "  POS (Singh et al. 2018 / LinCE): https://ritual.uh.edu/lince  -> CoNLL files, pass with --pos-train/--pos-test"
