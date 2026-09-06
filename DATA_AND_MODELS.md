# External data and model downloads

You do not need any download below to redraw the included figures, recompute
correlations or refit the two mappings from the packaged frozen feature grid.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for those commands.

## Raw recordings (optional)

Download from the original hosts and keep them outside the repository.
The sizes below are archive sizes in decimal units; extraction requires extra
disk space. The GSSC adapter extracts one EDF at a time into `--temp-dir`,
so it does not require unpacking the entire Kumral archive at once.

| Source/version | Original file | Bytes | Published MD5 |
|---|---|---:|---|
| [Kumral / FreiData v1](https://freidata.uni-freiburg.de/records/31mg4-mfq53) | `Kumral et al., 2023.zip` | 46,588,397,863 (46.6 GB) | `dc6d0768de5f521218d13c9768504e24` |
| [Zhang / figshare v1](https://doi.org/10.6084/m9.figshare.22226692.v1) | `Zhang & Wamsley 2019.zip` | 1,049,159,479 (1.05 GB) | `5854cfea4925f57d4d0a440518f4b72a` |

In each record page's Files section, choose Download for the ZIP. The project
historically renamed the Zhang archive `Zhang_Wamsley_2019_Final.zip`; its
contents are the original archive, not a different dataset. Optional resumable
macOS/Linux downloads (run only if you want the large files):

```bash
mkdir -p /absolute/path/to/tpmt_raw
curl -fL --retry 3 -C - \
  'https://freidata.uni-freiburg.de/api/records/31mg4-mfq53/files/Kumral%20et%20al.,%202023.zip/content' \
  -o '/absolute/path/to/tpmt_raw/Kumral et al., 2023.zip'
curl -fL --retry 3 -C - 'https://ndownloader.figshare.com/files/39504757' \
  -o '/absolute/path/to/tpmt_raw/Zhang_Wamsley_2019_Final.zip'
```

Replace `/absolute/path/to/tpmt_raw` with your chosen disk directory first.
On macOS run `md5 'path/to/archive.zip'`; on Linux use `md5sum`. Compare with
the table before processing. MD5 is supplied for archival integrity, not as
a security guarantee. Both datasets are CC BY 4.0; cite their investigators as
specified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The [DREAM metadata v9](https://doi.org/10.26180/22133105.v9) record provides
`Data records.csv`, `Datasets.csv` and `People.csv` (about 0.53 MB in total).
Download these if auditing original dataset/record IDs and endpoint labels.
The selected labels and record mapping needed here are already packaged;
the full database is not needed for the documented replay.

## GSSC 0.0.9 and pretrained weights (optional)

Official [source and instructions](https://github.com/jshanna100/gssc) and
[PyPI 0.0.9 wheel](https://pypi.org/project/gssc/0.0.9/).
The wheel is about 61.6 MB and includes both pretrained networks. After
installing the full dependencies in REPRODUCIBILITY.md, from the repository
root run:

```bash
python -m pip install --no-deps --target code/core/.vendor/gssc 'gssc==0.0.9'
python code/check_gssc_weights.py
```

The expected layout is:

```text
code/core/.vendor/gssc/
  gssc/infer.py
  gssc/nets/sig_net_v1.pt
  gssc/nets/gru_net_v1.pt
  gssc-0.0.9.dist-info/  (contains upstream metadata/license)
```

`--no-deps` avoids replacing the pinned full environment. Do not use it before
installing the dependencies. Use this pinned wheel rather than an unpinned
GitHub default branch when reproducing the archived model.

| File | SHA-256 |
|---|---|
| `gssc-0.0.9-py3-none-any.whl` (upstream release) | `e2ae56bf1ee581c5c0fe5b1481f7a424ff35a0029864749e1ffec79c0aca9976` |
| `sig_net_v1.pt` | `aa4bdcd7e29653138b096b3447d8db2d48508dc729e739b7ba9806dfd3c7c433` |
| `gru_net_v1.pt` | `124cfc858f49e3599b0798e494d7378d0d57a83d249faf79702da4cd8ee83304` |

These legacy checkpoints require PyTorch object deserialization in the
adapter. Obtain them only from the verified official release and verify the
hashes before loading; do not substitute untrusted `.pt` files. The adapter
runs CPU inference with the same preprocessing as the archived computation.
GSSC is AGPL-3.0, not MIT. The download is ignored by `.gitignore` and should
not be included in this lightweight repository.

## Reproducibility boundary

The portable JSON models and all archived derived data needed for replay are
included. The full historical grid-curation pipeline and fresh LLM prompting
are not a one-command raw-data rebuild in this release. Single-EDF physical
feature inference, raw Kumral staging, training-only refits from the frozen
grid, and result plotting have separate commands in the full guide. Do not
confuse a numerical replay with an independent replication on new data.
