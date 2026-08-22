# electoral-parser

OCR pipeline for India’s **2026 English SIR draft electoral rolls** (Telangana S29). Image-only PDFs → structured JSON/CSV, then CLI or local search UI.

Layout: **3×10 voter cards** per page (`eci-2026-en-3x10`). Grid-first detection; OpenCV contours as fallback. PaddleOCR English defaults to **PP-OCRv6_tiny**; invalid cards are re-OCR’d with **small** at end of each PDF (`--retry-small`). Use `--ocr-model small` to run small for everything. Wrapped Name / Father / Husband lines (4, 5, or 6 body lines) are continuation-parsed; missing House/Age/Relation get a second-pass OCR on detected ink bands, and **6-line wraps force fixed Age strips** (including a bottom crowded band). Sparse/last voter pages OCR a **padded full-card crop** (no photo wipe) so clipped serial/EPIC/Age survive; small-retry reuses kept page rasters when available.

Companion downloader: [electoral-downloader](https://github.com/MahmoodUlHassan/electoral-downloader).

## Setup

Python **3.12+** (3.13 is fine).

```bash
cd electoral-parser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # pytest
```

First OCR run downloads Paddle models to `~/.paddlex/official_models/` (one-time).

## Tests (no PDF required)

```bash
pytest -q
```

Layout tests against a sample PDF are skipped unless that file is present (path is in `tests/test_sample_layout.py`).

## Parse

```bash
python main.py --help
python main.py parse --help
```

PDFs are image JPEGs (~1983×2806). Native extract is the default (sharper than upsampling). Card OCR uses the **text + EPIC strip only** (photo wiped) at **2×**. Band refill runs only when first-pass fields are incomplete.

Point `--pdf` at a **file** or an **AC folder** of `part_*.pdf` (natural-sorted). Folder mode skips parts that already have `voters.csv` and appends to `output/csv/all_voters.csv`.

After a successful parse (≥90% of expected/extracted valid), **`--delete-pdf`** (default) removes the source PDF; **valid** card crops are deleted; **invalid** crops are kept; page rasters/overlays are kept only for pages that still have invalid cards (e.g. `debug/part_217/pages/page37.png`). Use **`--keep-pdf`** to retain the PDF. **`--workers 2`** (or 3) parses multiple PDFs in parallel (each process loads its own Paddle).

### Mini run — one voter page

~a minute after models are cached. Page 3 is the first full card grid on these rolls.

```bash
python main.py parse path/to/part_1.pdf --pages 3
```

With the sibling downloader:

```bash
python main.py parse \
  ../electoral-downloader/downloads/Rangareddy/52_Serilingampally/part_1.pdf \
  --pages 3
```

### Mini run — one full part

One PDF is typically **20–40 minutes**. AC 52 has **639** parts.

```bash
python main.py parse \
  ../electoral-downloader/downloads/Rangareddy/52_Serilingampally \
  --limit 1
```

Same thing as a single file:

```bash
python main.py parse \
  ../electoral-downloader/downloads/Rangareddy/52_Serilingampally/part_1.pdf
```

Optional debug overlays (keep off for batches):

```bash
python main.py parse path/to/part_1.pdf --pages 3 --visualize -v
```

Use `--profile` to print a phase timing table and write `logs/profile_<stem>.json`. Card OCR defaults to **`--ocr-workers 1`**. Primary model is **`--ocr-model tiny`**; after each PDF, invalid cards are re-read with **small** unless `--no-retry-small`. Header/footer OCR runs once per part, then reuses metadata.

Page subsets:

```bash
python main.py parse path/to/part_1.pdf --pages 3,14,22-30
```

### Full AC folder

Resume-safe: existing `output/part_N/csv/voters.csv` is skipped.

```bash
python main.py parse \
  ../electoral-downloader/downloads/Rangareddy/52_Serilingampally
```

Faster batch (2–3 workers; each needs RAM for Paddle):

```bash
python main.py parse \
  ../electoral-downloader/downloads/Rangareddy/52_Serilingampally \
  --workers 2
```

Keep PDFs after parse:

```bash
python main.py parse path/to/folder --keep-pdf
```

## Search and UI

Needs `output/csv/all_voters.csv` from parse.

```bash
python main.py search "Balamani"
python main.py search "SWD5889530"
```

Tokens are AND-matched across name, relative, EPIC, house, section, source PDF.

```bash
python main.py ui                 # http://127.0.0.1:8765
python main.py ui --port 8765
```

The UI reloads the combined CSV on each request (no restart after more parts parse). Table capped at 200 rows; count is the full hit size.

## Refill (no PDF re-OCR)

Second-pass House / Age / Relation from saved card crops under `debug/`. Use after a parse, or if wrap/house fields were missing.

```bash
python main.py refill
python main.py refill --json output/part_1/json/voters.json
```

## Output

| Path | Contents |
| --- | --- |
| `output/<stem>/json/voters.json` | Full records + `rawOcr` |
| `output/<stem>/csv/voters.csv` | One part |
| `output/csv/all_voters.csv` | Combined, with `sourcePdf` |
| `debug/<stem>/cards/` | Per-card crops |

CSV columns: `serialNo`, `epic`, `name`, `relationType`, `relativeName`, `houseNo`, `age`, `gender`, `page`, `partNo`, `constituency`, `section` (+ `sourcePdf` on the combined file).

`output/`, `debug/`, and `logs/` are gitignored.

## Notes

- **House number** is the door/plot field on the card. **Section** is the locality from the page header.
- Paddle’s detector often drops one of two stacked lines on a full-card OCR; band refill is what recovers House vs Age and wrapped names.
- Do not use ECI `acId` in download URLs — only `asmblyNo`.
