# Usage

## 1. Build local validation data

```powershell
python valid_builder.py --data-dir data_A --out-dir validation --max-valid 50000
```

Outputs:

- `validation/dataset1/train.csv`
- `validation/dataset1/valid.csv`
- `validation/dataset2/train.csv`
- `validation/dataset2/valid.csv`

`dataset2` uses `split=0` for train and `split=1` for validation.
`dataset1` uses a time split, with the latest 20% as validation by default.

Set `--max-valid 0` to use all validation positives.

## 2. Evaluate local MRR

```powershell
python evaluate_mrr.py --valid-dir validation
```

Output example:

```text
dataset1_mrr=0.12345678 rows=50000
dataset2_mrr=0.12345678 rows=50000
```

## 3. Try custom weights

Create a JSON file:

```json
{
  "dataset1": {
    "pair_count": 10.0,
    "pair_recency": 6.0
  },
  "dataset2": {
    "dst_popularity": 2.0,
    "dst_recent_popularity": 2.5
  }
}
```

Then run:

```powershell
python evaluate_mrr.py --valid-dir validation --weights weights.json
```

Only listed weights are overridden. Missing weights keep the defaults from
`rule_ranker_v2.py`.

## 4. Generate submission

Generate with the v2 rule ranker:

```powershell
python make_submission.py --data-dir data_A --out-dir submission --zip result.zip --ranker v2
```

Use custom weights:

```powershell
python make_submission.py --data-dir data_A --out-dir submission --zip result.zip --ranker v2 --weights weights.json
```

Use the old rule ranker for comparison:

```powershell
python make_submission.py --data-dir data_A --out-dir submission_v1 --zip result_v1.zip --ranker v1
```
