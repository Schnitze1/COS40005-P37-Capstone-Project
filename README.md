# CapEx 2026 Showcase Website — Project P37

**Project Title:** *Evaluating Trajectory Machine Unlearning and Backdoor Robustness in Offline Reinforcement Learning Applications*  
**Repository:** [`Schnitze1/COS40005-P37-Capstone-Project`](https://github.com/Schnitze1/COS40005-P37-Capstone-Project)  
**Branch:** `Website`  
**Deploy Target:** GitHub Pages (`gh-pages` branch)  

---

## 1. Architecture & Repository Isolation

This website is scaffolded with **Astro 7** and designed to present offline reinforcement learning research conducted on the **Swinburne OzSTAR supercomputing cluster**.

> [!IMPORTANT]
> **Branch Isolation Rule:**
> This website lives exclusively on the `Website` orphan branch. Active research code, heavy HDF5 datasets, PyTorch `.pt` checkpoints, and Slurm logs remain on research branches (`main`, `Alpha`, `Beta`). Never commit large binaries or datasets to this branch.

---

## 2. Project Structure

```text
D:\Repos\Capstone CapX Website\
├── .github/
│   └── workflows/
│       └── deploy-website.yml    # Automated GitHub Actions deployment to gh-pages
├── content/
│   └── site.json                 # Verified benchmark metrics, team bios, media placeholders
├── public/
│   ├── favicon.svg
│   └── media/                    # Media assets (once videos and renders are provided)
├── src/
│   ├── components/
│   │   ├── Nav.astro             # Global navigation header
│   │   ├── Footer.astro          # Global footer
│   │   └── sections/             # 10 Homepage section shells
│   │       ├── 00-Hero.astro
│   │       ├── 01-Problem.astro
│   │       ├── 02-FrankaKitchen.astro
│   │       ├── 03-Algorithms.astro
│   │       ├── 04-BafflePoisoning.astro
│   │       ├── 05-Unlearning.astro
│   │       ├── 06-ManiSkill2.astro
│   │       ├── 07-RoboDK.astro
│   │       ├── 08-Results.astro
│   │       └── 09-Team.astro
│   ├── layouts/
│   │   └── Layout.astro          # Base HTML layout & metadata
│   ├── pages/
│   │   ├── index.astro           # Cinematic homepage (10 sections)
│   │   ├── algorithms.astro      # Offline RL algorithm deep dive
│   │   ├── baffle.astro          # BAFFLE backdoor threat analysis
│   │   ├── unlearning.astro      # TrajDeleter unlearning demo
│   │   ├── maniskill.astro       # ManiSkill2 SAPIEN benchmark
│   │   ├── robodk.astro          # RoboDK digital twin simulation
│   │   ├── results.astro         # Complete results & divergence logs
│   │   └── team.astro            # Team P37 bios, supervisor & citations
│   └── styles/
│       └── global.css            # Dark research-cinematic tokens & utilities
├── astro.config.mjs
├── package.json
└── README.md
```

---

## 3. Local Development & Commands

Run all commands from the project root in PowerShell:

```powershell
# 1. Install dependencies
npm install

# 2. Start local development server (default: http://localhost:4321)
npm run dev

# 3. Build static production bundle to dist/
npm run build

# 4. Preview built production bundle locally
npm run preview
```

---

## 4. Deployment Pipeline

A dedicated GitHub Actions workflow (`.github/workflows/deploy-website.yml`) triggers on every push to the `Website` branch:
1. Checks out the `Website` branch.
2. Installs Node.js 22 dependencies via `npm ci`.
3. Compiles the static site via `npm run build` into `./dist`.
4. Deploys the `./dist` folder directly to the `gh-pages` branch using `peaceiris/actions-gh-pages`.

GitHub Pages should be configured in repository settings to serve from the root of the `gh-pages` branch.

---

## 5. Verified Metric Values (`content/site.json`)

All benchmark metrics in `content/site.json` reflect verified experimental outcomes:
* **BEAR Clean Baseline:** Lifetime Mean Return `1.67` across 1,000,000 steps without collapse (Job `16134221`, `opt1_1m_top4_lr1e6_s44`); Peak return `2.70` ("Best single evaluation (10 episodes)").
* **IQL:** Peak `3.50` at epoch 7 (also achieved at epoch 4); Mean of last 6 epochs: `2.62`.
* **CQL:** Peak `2.40` at epoch 8; Mean of last 6 epochs: `1.02`.
* **TD3+BC:** Peak `2.00` at epoch 1; Final epoch return `0.30`; Mean of last 6 epochs: `0.38`; Overall mean: `0.72`; Critic loss exploded from `228.38` to `145,720.22`.
* **Kitchen-partial KL:** Labeled `"1M final: 0.00 (both seeds)"`; Secondary callout: `"Best checkpoint: 1.10 at step 150k (Seed 42)"`.
* **BAFFLE Contiguous Poisoning:** Labeled `"Final kitchen task score (mean of last 10 evals)"` referencing clean BEAR baseline (`1.67`) across 1% (`1.55`), 5% (`1.53`), 10% (`0.86`), 15% (`1.30`), and 20% (`0.46`).
