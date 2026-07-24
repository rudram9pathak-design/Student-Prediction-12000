# AI Placement Intelligence Platform — integration notes

## What changed vs. the old app.py
Only presentation: layout, CSS, navigation, charts, and code organisation.
**No prediction math, feature engineering, or batch-scoring logic changed.**
`utils.py` holds every original model/data function unchanged — diff it
against your old `app.py` if you want to verify nothing moved.

## File layout
```
app.py            entry point — routing, forms, page bodies
theme.py          colour tokens + full CSS + animated background
components.py     header, nav pills, hero, KPI cards, badges, stepper
charts.py         Plotly charts (interactive) + the matplotlib PNG report
utils.py          model loading, schema, feature engineering, batch defaulting
requirements.txt
.streamlit/config.toml
assets/           drop a licensed logo here if you have one (see below)
```

## To run it
Put these next to `app.py` (same as before — nothing new required here):
- `model.pkl`
- `feature_schema.json`

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Things worth knowing
- **Logo:** the brief asked for the "official Grant Thornton Bharat logo."
  I didn't have a licensed asset to embed, so the header uses a small
  original abstract wordmark instead of the real trademark — drop a real
  logo file into `assets/` and swap the inline SVG in `components.py::header()`
  for an `st.image()` call if you have licensing rights to use it.
- **Charts:** all on-screen charts (gauges, bars, radar, histogram, donut)
  are now Plotly. The one exception is the *downloadable* full-report PNG —
  that's still matplotlib, on purpose, so you don't need a headless-browser
  PNG exporter (`kaleido`) just for one export button. Nothing about what
  that button produces has changed.
- **Testing:** every page and both prediction branches (placed / at-risk),
  plus a batch CSV upload with missing columns, were smoke-tested end-to-end
  with `streamlit.testing.v1.AppTest` against a dummy model/schema — zero
  exceptions. You'll still want to click through it once with your real
  `model.pkl` before shipping.
