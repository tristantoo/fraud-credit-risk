# Freddie Mac data: steps for Tristan (part b)

Part b needs the Freddie Mac Single-Family Loan-Level Dataset, which requires your own free registration. Nothing in
this repo has touched real Freddie data. Checked against Freddie Mac's site on 2026-10-23 (Release 47, July 2026).

## Layout: updated to Release 47

Freddie Mac's July 2026 files have 31 origination fields and 35 performance fields, and the performance files are
named `sample_perf_YYYY.txt`. The code was updated to match on 2026-10-23, before any real data was downloaded
(PLAN.md deviation 8). The column positions it reads did not move. If you download a later release, check its File
Layout against `ORIG_NFIELDS, SVCG_NFIELDS` in `src/frc/freddie.py` first; `check_layout` stops on a mismatch.

## 1. Register

1. Open https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset and read the Terms and Conditions linked
   there (free for non-commercial, academic and research use; no redistribution of the raw files).
2. Click **Access Historical Data**, which goes to Clarity Data Intelligence
   (https://claritydownload.fmapps.freddiemac.com/CRT/). Choose the option to register a new account, fill in the
   form with your own details, confirm the email, and sign in. (Clarity's sign-in page could not be inspected from
   here, so the button labels may differ slightly.)
3. Accept the dataset terms when prompted and open the **SFLLD Data Download** section.

## 2. Download

From the download page, the **Sample** dataset (a simple random sample of 50,000 loans per full vintage year):

- `sample_2005.zip` through `sample_2016.zip` at minimum (the pre-registered range in PLAN.md, "Tristan's steps").
  Each zip holds `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`, pipe-delimited, no header row.
- Optional: later vintages (2017 onward). They make the out-of-time test population look more like a current
  portfolio. Decide before looking at any result and write the chosen range into PLAN.md.

From the public dataset page (no login):

- General User Guide, July 2026: https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf
- File Layout, July 2026: https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx
- File headers: https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip
- The Release Notes (linked from the guide) for the origination and performance cutoff dates.

Do not download the full `historical_data_YYYYQn.zip` files; the pipeline is built for the sample and the full set is
far larger.

## 3. Put the files in place

Unzip so the `.txt` files sit anywhere under `data/raw/freddie/` (the script searches recursively). `data/raw/` is
gitignored, so the raw data never enters git.

```powershell
cd path\to\fraud-credit-risk
New-Item -ItemType Directory -Force data\raw\freddie
Get-ChildItem "$HOME\Downloads\sample_20*.zip" | ForEach-Object { Expand-Archive $_.FullName -DestinationPath data\raw\freddie -Force }
Get-ChildItem data\raw\freddie -Recurse -Filter *.txt | Select-Object Name, Length
```

You should see one `sample_orig_YYYY.txt` and one `sample_perf_YYYY.txt` per vintage.

## 4. Verify the layout against the guide

Open the File Layout xlsx and confirm, for the release you downloaded, that origination positions 0, 1, 2, 5, 7 to
13, 17, 19, 20, 22 and performance positions 0 to 4, 8 and 10 (0-based) are the fields named in
`src/frc/freddie.py` (`ORIG` and `SVCG` dicts). For Release 47 they are; only the field counts and the file name
changed (see the top of this page). A quick check of the real counts:

```powershell
Get-Content data\raw\freddie\sample_orig_2005.txt -TotalCount 1 | ForEach-Object { ($_ -split '\|').Count }
Get-Content data\raw\freddie\sample_perf_2005.txt -TotalCount 1 | ForEach-Object { ($_ -split '\|').Count }
```

## 5. Run

After the code change above (and a passing `uv run pytest`):

```powershell
uv run python scripts/run_part_b.py
```

It writes `outputs/part_b_vintage_curves.csv`, `part_b_roll_rates.csv`, `part_b_pd_metrics.csv`,
`part_b_pd_reliability.csv` and `part_b_pd_contrasts.csv`, and prints the metrics and contrasts. Untested on real
data: expect each vintage's performance file to load whole into memory (one at a time) and the 5,000-resample
bootstrap over test loans to take tens of minutes.

Then: read the results against PLAN.md b1 (H5: LR out-of-time AUC CI excludes 0.5; H6: calibration-in-the-large CI
contains 0, which may well fail across a regime change; GBM vs LR Holm-adjusted), write them up as they are, and
commit the outputs (never the raw files).
