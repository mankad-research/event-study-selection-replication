# Data

`working_data_public.csv` is the analysis file for the application in Section 5 of the paper. It has one row per
executive appointment (n = 3,103), of which 1,574 were covered by the *Wall Street Journal*. It contains only the
variables the replication code uses; firm and security identifiers, names, and all other fields are excluded.

| Column | Description |
|---|---|
| `event_id` | Arbitrary event identifier (1 to 3,103). |
| `abret_10` | Abnormal return (percent) around the press-release announcement date. The outcome for every event. |
| `C` | 1 if the appointment was covered by the *Wall Street Journal*, 0 otherwise. |
| `Female` | 1 if the appointee is female. |
| `logMV` | Log market value of the firm. |
| `Outsider` | 1 if the appointee was hired from outside the firm. |
| `Analyst` | log(1 + number of analysts following the firm). |
| `wsj_lag_days` | Days from the press release to the *Wall Street Journal* article, for covered events (blank otherwise). Used only for the article-timing facts reported in Section 5. |

## Notes

- Keep the rows in their current order. The supplementary samples in `run_realdata.py` are drawn by row position,
  so sorting or filtering the file changes the draws.
- `run_realdata.py` reads the file with `float_precision="round_trip"`, which recovers the stored values exactly.
  Other readers may differ in the last digits, which changes results only at the level of rounding error.
