# Processed simulation inputs

Only the inputs read by the IEEE33 environment are included. These are the
processed values used by the selected case; no synthetic substitute is used.

- **Ausgrid Solar Home Electricity Data:** 128 paired household load/PV profiles,
  July 2010–June 2013. Original half-hour interval energy is converted to power,
  filtered and interpolated to 15 minutes. Load and PV are normalized by the
  original per-household 99th and 99.5th percentile power, respectively.
- **National Grid Electric Nation, Green Flux 15 Minute:** 57,054 reconstructed
  charging sessions, June 2017–December 2018. Energy is reconstructed from
  measured average current at nominal 230 V; it is not measured battery SOC or
  requested energy. Arrival/departure timestamps retain the original processed
  local wall-clock convention.
- **AEMO VIC1 prices, 2019:** interpolated to 15 minutes. Retail price is
  `0.2 + 0.8 * clip(spot_AUD_per_MWh, -100, 500) / 1000` AUD/kWh.
- `splits.json` retains the train/validation/test partitions and fixed scenarios.

Profile columns and EV record identifiers use neutral labels. Unused EV columns,
raw source archives, and source metadata files are omitted. Numeric inputs,
timestamps, row/column ordering and split definitions are preserved. Interpolation
and reconstruction do not create additional measured information. The simulator
uses these inputs to construct feeder scenarios and the configured EV schedules.

These processed datasets retain their original source terms; this package does
not assign a new license to the data.
