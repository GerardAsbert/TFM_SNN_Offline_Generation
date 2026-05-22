# Handwritten Text Generation with Spiking Neural Networks (SNNs)

Authors: Marc Pomar Pallarès, Aksel Serret, Xavier Otazu, Alicia Fornes, Gerard Asbert

> Data-efficient generation of multi-stroke Japanese characters using spiking neural networks trained with eligibility propagation (e-prop). We extend the classic NEST “chaos” tutorial with:
> - (1) multi-letter learning via per-letter frozen inputs,
> - (2) an explicit pen-lift channel, and
> - (3) controlled temporal jitter for natural variation — all from a single exemplar per character.

---

## Highlights
- **Single exemplar per class** for training (data efficiency).
- **Multi-letter training** in one network (implicit labeling via frozen inputs).
- **Pen-lift readout** for clean multi-stroke characters.
- **Temporal jitter** during training/inference for human-like variability.
- **Different styles** for handwriting variability.

---

## Repo Structure (placeholder)
```bash
.
├─ data/
│  ├─ raw_svg/           # KanjiVG or other SVG sources
│  ├─ processed_txt/     # Δx, Δy, pen_flag per character (fixed length)
│  └─ examples/          # small sample set for quick tests
├─ images/               # figures for README/paper/slides
├─ src/
│  ├─ data_prep/         # svg_to_traj.py, resampling, normalization
│  ├─ snn/               # build_network.py, eprop_utils.py, jitter.py
│  ├─ train/             # train_multiletter.py
│  └─ infer/             # infer_letters.py, plot_outputs.py
├─ configs/
│  ├─ train_hiragana.yaml
│  └─ infer.yaml
├─ env/
│  └─ environment.yml    # conda env (Python + NEST + libs)
├─ paper/
│  └─ (optional) manuscript or slides
└─ README.md
```
---

## Requirements
- Python 3.10+
- NEST Simulator 3.x
- NumPy, SciPy, Matplotlib, Pandas
- svgpathtools (for SVG parsing)
- PyYAML, tqdm

### Quick install (conda)
```bash
conda env create -f env/environment.yml
conda activate snn-handwriting
```
If you prefer pip:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Install NEST per your OS docs (may require system packages)
```

⸻

## Data
Initial processing of images:
-	We use KanjiVG (or any SVG source) for stroke paths.
-	Convert SVG to fixed-length sequences of Δx, Δy, pen_flag (e.g., 50 points at 8 ms each → 400 ms sequence).

Convert SVG to training txt
```bash
python src/data_prep/svg_to_traj.py \
  --input data/raw_svg/03042.svg \
  --output data/processed_txt/a.txt \
  --samples_per_stroke 200 \
  --pen_up_pause 40 \
  --max_points 50
```

⸻

## Training

Train multi-letter SNN with e-prop. Letters are distinguished by per-letter frozen Poisson inputs (implicit labels). A third readout learns the pen-lift channel.
```bash
python src/train/train_multiletter.py \
  --letters あ い を や き \
  --txt_dir data/processed_txt \
  --n_rec 400 \
  --n_iter 500 \
  --dt_ms 0.125 \
  --data_point_ms 8 \
  --train_jitter_ms 0.5 \
  --save_path outputs/run_001/
```
Outputs:
- Trained weights (pickle)
- Loss curves (MSE on x,y; pen-lift accuracy)
- Readout traces

⸻

## Inference

Rebuild the network with learning disabled, load the trained weights, and generate trajectories. You can add inference-time jitter for stylistic variation.
```bash
python src/infer/infer_letters.py \
  --letters を ぬ や き ん \
  --weights outputs/run_001/weights.pkl \
  --infer_jitter_ms 2.0 \
  --plots_dir outputs/run_001/infer_plots/
```

⸻

## Key Ideas
(No tinc molt clar que posar)
-	Frozen inputs per letter: deterministic Poisson spike trains (seeded by Unicode) serve as an implicit class label.
-	Pen-lift neuron: third readout learns when to draw (pen down) vs. not (pen up).
-	Jitter: small time shifts (ms) during training act as data augmentation; at inference, they produce natural variation.

⸻

## Results
(Poso les del meu treball pero les haurem de modificar)
-	images/loss_convergence.png — learning curves: 1 vs 2 vs 5 letters.
-	images/penlift_before_after.png — effect of pen-lift channel on multi-stroke characters.
-	images/jitter_grid_ta.png — 4x4 grid (train jitter vs inference jitter).
-	images/mse_loss_letters.png — per-letter MSE over epochs.

⸻

## Reproducibility

Placeholder tambe:
  
- RNG seeds fixed per letter using Unicode code point (e.g., seed = ord(letter) * 1000).
- All signals resampled to fixed length (max_points) and fixed time step (data_point_ms).

⸻

## Citation

If you use this repo, please cite:

- The repository (once a paper/DOI is available)
- NEST Simulator
- KanjiVG

⸻

## License

TBD

⸻

## Contact

**Authors**

- Marc Pomar Pallarès — (marcpomar.cvb@gmail.com)
- Aksel Serret — (akselserret@gmail.com)
- Xavier Otazu — (xotazu@cvc.uab.cat)
- Alicia Fornes — (afornes@cvc.uab.es)
- Gerard Asbert - (email)


Feedback and PRs are welcome.
