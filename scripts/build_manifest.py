#!/usr/bin/env python3
"""
Build manifest CSV with columns: name,feature_path,translation

Usage:
  python scripts/build_manifest.py --corpus_csv annotations/manual/PHOENIX-2014-T.train.corpus.csv \
      --features_root data/features/train --out manifest_train.csv --split train
"""
import os
import pandas as pd
import argparse
import glob

def main(corpus_csv, features_root, out_csv):
    # read corpus: header name|video|start|end|speaker|orth|translation
    df = pd.read_csv(corpus_csv, sep='|', dtype=str)
    records = []
    for _, row in df.iterrows():
        name = row['name']
        feat_path = os.path.join(features_root, name + '.npy')
        if not os.path.exists(feat_path):
            # try alternative path structures
            matches = glob.glob(os.path.join(features_root, '**', name + '.npy'), recursive=True)
            if matches:
                feat_path = matches[0]
            else:
                # skip missing
                print(f"Missing feature for {name}, expected at {feat_path}")
                continue
        translation = row.get('translation', '') if 'translation' in row else row.get('orth', '')
        records.append({'name': name, 'feature_path': feat_path, 'translation': translation})
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote manifest with {len(out_df)} entries to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_csv', required=True)
    parser.add_argument('--features_root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.corpus_csv, args.features_root, args.out)
