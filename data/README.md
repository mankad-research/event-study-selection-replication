# Data

`working_data_dedup.pkl` (pandas pickle) and `working_data_dedup.csv` (identical contents) are the analysis file for
Section 4 of the paper. Each row is one top-management-team appointment (Chairman, CEO, or President) at a U.S. public
company, announced by press release between February 1999 and January 2009. There are 3,103 rows, of which 1,574 were
also covered by the Wall Street Journal. The scripts read the `.pkl`; the `.csv` is provided for use outside Python.

## Source

The appointments come from the dataset of Allen and Schmidt (2025), "Event study misestimation and discretionary media
reporting" (working paper, SSRN 4872176). That dataset was built by text analysis of Wall Street Journal articles and
company press releases, manual coding of their content, and fuzzy matching of the events to firm-level data from CRSP,
Compustat, Thomson Reuters, and IBES.

## Variables

| Column | Description |
|---|---|
| `abret_10` | Abnormal return (percent) around the press-release announcement date. Used as the outcome for every event. |
| `C` | 1 if the appointment was covered by the Wall Street Journal, 0 if it was disclosed only by press release (1,574 covered). |
| `Female` | 1 if the appointee is female. |
| `logMV` | Log of the firm's market value. |
| `Outsider` | 1 if the appointee is from outside the firm, 0 for an internal promotion. |
| `Analyst` | log(1 + number of analysts following the firm). |

The file has no missing values.

## What is not included

Firm and appointee names, firm identifiers (e.g., gvkey, permno, CUSIP, IBES ticker), event dates, and the other
vendor-provided fields in the source dataset are not distributed. The six columns above are the only ones the
programs in `code/` use. An earlier version of the file also contained a `returns` column that mixed returns from two
different event windows (the WSJ-article date for covered events, the press-release date for uncovered events); it is
not used and is not included.

`prep_data_original.py` shows how the pre-deduplication file was built from the source `.dta` file, which is not
distributed.

## Construction notes

- **One row per firm-event.** Some press releases announce several appointees on the same date, with identical returns.
  These are collapsed to one observation. The file contains no duplicate firm-and-press-release-date pairs.
- **Abnormal returns.** `abret_10` is computed with the market-model procedure described in Allen and Schmidt (2025):
  firm returns are regressed on market returns over a benchmark period before the event, and the abnormal return is
  the actual return minus the prediction, summed over the event window.

## Reference

Allen, A., and Schmidt, W. (2025). Event study misestimation and discretionary media reporting. Working paper.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4872176
