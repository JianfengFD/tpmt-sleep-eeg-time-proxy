# External GSSC dependency

The GSSC source tree and pretrained weights are intentionally not bundled.
Place or symlink the public GSSC checkout at `code/core/.vendor/gssc` before
running `analysis.sleep_staging.kumral_gssc_rem_probability`.

The required files are:

- `gssc/nets/sig_net_v1.pt`, SHA-256
  `aa4bdcd7e29653138b096b3447d8db2d48508dc729e739b7ba9806dfd3c7c433`
- `gssc/nets/gru_net_v1.pt`, SHA-256
  `124cfc858f49e3599b0798e494d7378d0d57a83d249faf79702da4cd8ee83304`

The analysis used GSSC 0.0.9 under its AGPL-3.0 license. See the top-level
`REPRODUCIBILITY.md` for the installation and inference command.
