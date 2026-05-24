# k2dex · science

Interactive tools and explainers applying complexity-science methods to competitive Pokémon (VGC) team composition data.

The core model is a pairwise maximum-entropy (inverse Ising) fit on tournament team rosters from [Limitless VGC](https://play.limitlesstcg.com/). The fitted couplings *J* and biases *h* capture which Pokémon (and held items) tend to appear together on teams — and which ones compete for the same slot. The webapp surfaces this as a team completer, per-team diagnostics, and format-wide statistics, plus an interactive [Science](https://kylehtunis.github.io/k2dex-science/science) page explaining the math from first principles.

**[Live site →](https://kylehtunis.github.io/k2dex/)**

## Running locally

### Static webapp (React/TypeScript)

```bash
cd web
npm install
npm run dev          # http://localhost:5173
npm test             # vitest suite
npm run build        # production build → web/dist/
```

### Streamlit app + notebooks (Python)

```bash
pip install numpy scipy scikit-learn streamlit matplotlib
streamlit run scripts/app.py
jupyter lab
```

### Tests

```bash
python -m unittest discover tests
```

## License

[GPL-3.0](LICENSE)
