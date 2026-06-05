# Project Folder Structure

```
JacksonInvestments/
├── backend/
│   ├── core/          # Shared code used by both training and inference
│   ├── training/      # Offline model training pipeline
│   ├── advisor/       # FastAPI + LangChain advisor app (built later)
│   ├── notebooks/     # Jupyter notebooks for exploration
│   └── tests/         # Unit tests
│
├── frontend/          # Next.js chat UI (built later)
│
├── saved_models/      # Trained model weights — gitignored
│   ├── lstm/
│   └── rl/
│
├── data/              # Cached market data — gitignored
│   ├── raw/
│   └── processed/
│
└── docs/              # Project documentation and design artefacts
    └── diagrams/      # draw.io diagram source files
```

---

## Notes

**`core/` is shared** because the same feature engineering code runs during training and at inference time. Keeping it in one place means training and the live advisor always compute the same indicator values for the same input.

**`data/` and `saved_models/` are gitignored** because they hold large files that change every training run. The `.gitkeep` files in each folder preserve the structure in git without committing the contents.

**`advisor/` is empty for now.** I will build it after the training pipeline is complete and the models are validated through backtesting.
