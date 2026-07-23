# Research memo — synthetic render vs hand-label for deer 2D pose

**2026-07-23.** Every load-bearing number extracted from the primary PDF unless
labelled otherwise. **VERIFIED** / **REPORTED** / **UNVERIFIED**.

Distilled into `docs/CV_PIPELINE_DESIGN.md`. This file keeps the evidence.

---

## 1. Bottom line

**Hobby-mesh fidelity does not sink the synthetic approach — because the premise
is wrong in two ways, and the synthetic path is dominated by a cheaper
alternative regardless.**

**Finding A — the "professional CAD assets" premise is false. [VERIFIED]**
Mu et al.'s footnote points at
`unrealengine.com/marketplace/en-US/product/animal-pack-ultra-01` = *Animal Pack
Ultra 2*: **horse 3,299 vertices, tiger 3,124 vertices**, 2048² textures, 4–9
canned animations, **no LODs**. A ~$50 indie game pack — no fur cards, no
subsurface scattering, no PBR authoring. The CVPR 2020 result that founded this
line of work used assets plausibly *worse* than a competent hobby rig.

**Finding B — mesh fidelity is not what makes synthetic-only fail.** Synthetic-only
is catastrophically weak across wildly different asset tiers:

| Source | Asset / renderer | Synthetic-only PCK@0.05 |
|---|---|---|
| Mu et al. 2020 | UE4 game pack, Unreal | **25.33** horse / **21.17** tiger |
| SPAC-Net "Simple" | CGTrader model, Blender | **36.7** zebra / **29.9** rhino |
| PASyn (SynAP) | + VAE poses + style transfer | **46.6** zebra / **64.9** rhino |
| ZebraPose 2024 | Sketchfab model, photoreal sim | **46.6** zebra |

Upgrading from "game asset on COCO" to "photorealistic simulator" moves
synthetic-only ~25 → ~47. **99 hand-labelled real zebra images give 78.7.**

**Finding C — the fidelity question is genuinely unmeasured for animal pose**, but
a good proxy exists (SPAC-Net Table 3) and it favours deer. See §3.

### Recommendation

1. **Hand-label 100–200 real deer frames at a URDF-aligned keypoint set and
   fine-tune.** A few hours; on the evidence it beats the whole synthetic + DA
   pipeline.
2. **Bootstrap labelling with SuperAnimal-Quadruped zero-shot**, then correct.
3. **Keep the renderer, repurposed** — as the IK fitting model, and as
   depth/edge conditioning for a diffusion appearance upgrade if scale is ever
   needed.
4. **Do not invest in fur, subsurface scattering, or material authoring.**

---

## 2. The Mu et al. baseline

CVPR 2020 oral. https://arxiv.org/abs/1912.08265 ·
https://openaccess.thecvf.com/content_CVPR_2020/papers/Mu_Learning_From_Synthetic_Animals_CVPR_2020_paper.pdf
· https://github.com/JitengMu/Learning-from-Synthetic-Animals

### Recipe [VERIFIED — §4.1]

| Parameter | Value |
|---|---|
| Asset | **Purchased UE4 Marketplace pack** (Animal Pack Ultra 2) |
| Renderer | Unreal Engine + UnrealCV |
| Camera | 640×480, FOV 90°; network crops 256×256 |
| Images / animal | **10,000** = 5,000 random-texture + 5,000 original-texture |
| Split | 4:1 → 8,000 train / 2,000 val |
| Texture randomisation source | **COCO val2017** applied to the animal |
| Background | **COCO val2017** |
| Lighting / viewpoint | Randomised |
| Pose source | **Random frames from the pack's canned animations** (4–9 per animal) |
| Extra GT | Depth, 9-part segmentation, dense 2D+3D keypoints |
| Backbone | Stacked Hourglass, RMSProp, 100 epochs |

### Table 1 — PCK@0.05 on TigDog [VERIFIED verbatim]

**Horse** (Eye / Chin / Shoulder / Hip / Elbow / Knee / Hoove / **Mean**):

| Method | | | | | | | | Mean |
|---|---|---|---|---|---|---|---|---|
| **Syn (synthetic-only)** | 46.08 | 53.86 | 20.46 | 32.53 | 20.20 | 24.20 | 17.45 | **25.33** |
| CycleGAN | 70.73 | 84.46 | 56.97 | 69.30 | 52.94 | 49.91 | 35.95 | 51.86 |
| BDL | 74.37 | 86.53 | 64.43 | 75.65 | 63.04 | 60.18 | 51.96 | 62.33 |
| CyCADA | 67.57 | 84.77 | 56.92 | 76.75 | 55.47 | 48.72 | 43.08 | 55.57 |
| **CC-SSL** | 84.60 | 90.26 | 69.69 | 85.89 | 68.58 | 68.73 | 61.33 | **70.77** |
| Real (supervised) | 79.04 | 89.71 | 71.38 | 91.78 | 82.85 | 80.80 | 72.76 | **78.98** |
| CC-SSL-R (syn+real) | 89.39 | 92.01 | 69.05 | 92.28 | 86.39 | 83.72 | 76.89 | **82.43** |

**Tiger:**

| Method | | | | | | | | Mean |
|---|---|---|---|---|---|---|---|---|
| **Syn** | 23.45 | 27.88 | 14.26 | 52.99 | 17.32 | 16.27 | 19.29 | **21.17** |
| CycleGAN | 71.80 | 62.49 | 29.77 | 61.22 | 36.16 | 37.48 | 40.59 | 46.47 |
| BDL | 77.46 | 65.28 | 36.23 | 62.33 | 35.81 | 45.95 | 54.39 | 52.26 |
| CyCADA | 75.17 | 69.64 | 35.04 | 65.41 | 38.40 | 42.89 | 48.90 | 51.48 |
| **CC-SSL** | 96.75 | 90.46 | 44.84 | 77.61 | 55.82 | 42.85 | 64.55 | **64.14** |
| Real | 96.77 | 93.68 | 65.90 | 94.99 | 67.64 | 80.25 | 81.72 | **81.99** |
| CC-SSL-R | 95.72 | 96.32 | 74.41 | 91.64 | 71.25 | 82.37 | 82.73 | **84.00** |

### Reading them

- **Synthetic-only = 25.33 / 21.17.** Unusable. Note the structure: eye 46.08,
  chin 53.86, **hooves 17.45** — synthetic-only finds high-contrast head features
  and fails at limb joints, exactly the keypoints a URDF fit needs.
- **CC-SSL does essentially all the work: +45.4 (horse), +43.0 (tiger).**
  Attributing the residual gap to mesh quality is unsupported.
- Multi-task part-segmentation auxiliary was worth **+1.4 pts** (Table 3:
  60.84/50.26 → 62.25/51.69). [VERIFIED]
- **No texture-realism ablation exists** — they generated half random-texture and
  half original but never reported the conditions separately. [VERIFIED; the CVPR
  supplementary 403s to automated fetch, so "none in supplementary" is REPORTED
  via ar5iv]

**Licensing:** repo is **GPL-3.0** [VERIFIED]. TigDog is **non-commercial research
only** and the creators do not own the source video copyright [REPORTED].

---

## 3. Realism vs randomisation — how the tension resolves

### 3.1 The decisive experiment: SPAC-Net Table 3 [VERIFIED]

https://arxiv.org/abs/2305.17845 — holds 3D model and pose distribution fixed,
varies **only the appearance pipeline**. Closest thing to a controlled fidelity
ablation in animal pose.

**Synthetic-only, PCK@0.05, HRNet-w32:**

| Appearance pipeline | Zebra-300 | Rhino-300 |
|---|---|---|
| Simple (model on random background, original texture) | 36.7 | 29.9 |
| SynAP (+ VAE poses + StyTr² style transfer) | 46.6 | 64.9 |
| ControlNet (generative photorealism) | 70.9 | 65.8 |
| Bi-ControlNet (SPAC-Net) | **80.4** | **71.5** |
| *Reference: 99 real labelled images, no synthetic* | *78.7* | *88.3* |

**With 99 real labelled images added (Table 1):**

| Training set | Zebra-300 | Rhino-300 |
|---|---|---|
| R(99) only | 78.7 | 88.3 |
| R(99) + AP10K real 8K | 91.4 | 96.7 |
| R(99) + SynAP 3K (crude/styled) | 92.4 | 95.9 |
| R(99) + SPAC 3K (photoreal) | **96.3** | **95.9** |

**Three conclusions:**

1. **Appearance fidelity is worth a lot with zero real labels and almost nothing
   with ~100.** Zebra +43.7 synthetic-only (36.7→80.4) but only **+3.9** with 99
   real (92.4→96.3). Rhino: **+0.0** (95.9→95.9). *That is the resolution of the
   realism-vs-randomisation tension — both camps are right, in different regimes.*
2. **Coat type determines how much realism buys.** Zebra is a *texture-identity*
   animal; rhino is *shape-defined*. Crude→photoreal bought zebra **+33.8**
   (46.6→80.4) and rhino **+6.6** (64.9→71.5). **A deer is a rhino, not a zebra**
   — plain brown coat, no diagnostic surface pattern, silhouette-defined. The
   evidence specifically predicts surface realism matters *less* for deer.
3. **Crude synthetic + 99 real beats 8,000 real** (zebra 92.4 vs 91.4). A crude
   renderer is a supported *augmentation*, not a supported *substitute*.

### 3.2 BEDLAM's realism claim is weaker than its reputation [VERIFIED]

https://arxiv.org/abs/2306.16940

- **The clothing-realism claim is prose only. There is no table.** The paper says
  realistic clothing "leads to significantly better results than training with
  textured bodies" — with **no MPJPE/PA-MPJPE/PVE numbers**, in main paper or
  supplementary (checked main text, HTML, ar5iv). **The fidelity variable is not
  isolated with numbers anywhere in BEDLAM.**
- **The paper's own caveat:** the clothing effect is *"more pronounced when using
  a backbone pre-trained on ImageNet rather than COCO"* — i.e. **a backbone
  already pretrained on 2D pose absorbs most of the realism deficit.** The most
  actionable sentence in BEDLAM for us.
- What BEDLAM *does* quantify (Tables 3/6) is **backbone pretraining**, and it
  dwarfs everything: from scratch **61.8** → ImageNet **51.8** → **COCO 47.4**
  PA-MPJPE.

### 3.3 The domain-randomisation counter-position, quantified

Eversberg & Lambrecht, Sensors 2021, *Realism versus Domain Randomization* —
https://pmc.ncbi.nlm.nih.gov/articles/PMC8659618/ — vary one factor at a time,
AP[0.5:0.95]:

| Factor | Variant | AP |
|---|---|---|
| **Lighting** | white point lights | 0.623 |
| | temperature-varied | 0.635 |
| | **HDRI image-based** | **0.642** |
| **Background** | **random COCO images** | **0.642** |
| | HDRI images | 0.589 |
| | realistic deployment-domain | 0.612 |
| **Object texture** | grey base colour (none) | 0.642 |
| | random COCO images | 0.623 |
| | random material textures | 0.644 |
| | **realistic material textures** | **0.653** |
| | real scanned material | 0.648 |
| **Distractors** | none | 0.653 |
| | realistic YCB tools | 0.660 |
| | **random-textured cubes** | **0.669** |

**The entire object-texture swing is 3.0 AP, and flat grey (0.642) is within 1.1
of the best realistic material (0.653). Background choice swings 5.3, and
randomised backgrounds beat realistic ones.** Randomisation wins for context;
mild realism wins for the object; the object-surface term is the smallest lever.

⚠️ Rigid industrial object detection, not articulated animal keypoints —
analogically useful, not conclusive.

**Disconfirming evidence, actively sought:** Hodaň et al. 2019
(https://arxiv.org/abs/1902.03334) report **+24 % absolute mAP@.75 on Rutgers APC
and +11 % on LineMod-Occluded** from full PBR vs objects on random photographs
[REPORTED]. A large realism win — but the task is *instance* detection of specific
rigid objects, where surface appearance **is** class identity. The zebra case, not
the deer case.

### 3.4 Texture randomisation → shape bias: measured [VERIFIED]

- Geirhos et al. ICLR 2019 (https://arxiv.org/abs/1811.12231): ImageNet CNNs are
  texture-biased; Stylized-ImageNet induces shape bias [REPORTED].
- **Thalhammer et al. 2024** (https://arxiv.org/abs/2402.04878) [VERIFIED, Table 3]
  — classification of **textureless** objects:

| Training regime | Top-1 | Top-5 |
|---|---|---|
| texture (standard) | 63.22 | 86.27 |
| shape via style transfer (Geirhos) | 75.88 | 91.16 |
| **shape via UV-mapped random textures (render-time)** | **87.27** | **97.37** |

  Their stated mechanism is exactly the hypothesis: random-texture UV mapping
  *"provides geometrically accurate visible cues for observing the object
  geometry"* and *"encoded representations are agnostic to texture."*
  **Random-texture rendering induces shape bias better than style transfer, and
  it is the cheap option.**

### 3.5 Disconfirming evidence on random textures — take seriously

- **PASyn explicitly criticises Mu et al.'s scheme** [VERIFIED]: random ImageNet
  textures *"further increases the discrepancy between synthetic and real domains
  and thus weakens the improvement of synthetic data."* Their style-transfer
  alternative outperformed it.
- **ZebraPose (2024)** found vanilla photorealistic renders **outperformed**
  SPAC-Net's style-transferred data by *"3 to 25%"* [REPORTED] — the opposite
  direction.

**Honest resolution: the appearance-treatment literature is inconsistent at the
margins.** What is *not* inconsistent: synthetic-only is weak everywhere; the
appearance term collapses once real labels exist; shape/silhouette carries
keypoint localisation.

### 3.6 Silhouette vs surface, and skeleton accuracy

- **STRAPS** (https://arxiv.org/abs/2009.10013) deliberately feeds **silhouettes
  and 2D joints** as proxy representations, trained purely on synthetic,
  *specifically to sidestep the appearance gap*. Photorealism is designed out.
  [REPORTED]
- *Learning Part Segmentation from Synthetic Animals* (WACV-W 2024,
  https://arxiv.org/abs/2311.18661): part segmentation is *"transferable among
  species of similar structures regardless of shape and texture difference"*;
  synthetic tiger+horse transfers across **46 quadruped species**. [REPORTED]
- **Skeleton/pose coverage is the binding limitation, not appearance** [VERIFIED
  full-text]: *"the animation sequences for each CAD animal model provide limited
  poses"*; failure analysis is *"Due to the limited poses contained in CAD models,
  the model fails to segment the torso out."* Adding SMAL-generated poses gave
  +1.6 mIoU (40.61→42.21) and +2.59 (55.45→58.04).

**No clean silhouette-vs-surface ablation for animal keypoints exists
[UNVERIFIED], but every indirect line points the same way: skeletal structure and
pose coverage dominate; surface appearance is secondary; an inaccurate
skeleton/proportions is far more damaging than an inaccurate coat.** Our asset
profile (accurate silhouette, crude surface) is the favourable half.

---

## 4. How much does domain adaptation rescue a weak renderer?

### UDA-Animal-Pose, CVPR 2021 Oral [VERIFIED]
https://arxiv.org/abs/2103.14843 · https://github.com/chaneyddtt/UDA-Animal-Pose
(**MIT** [VERIFIED])

| Method | Horse | Tiger |
|---|---|---|
| Syn (synthetic-only) | 25.33 | 21.17 |
| CC-SSL (Mu et al.) | 70.77 | 64.14 |
| **MDAM-MT (theirs)** | **79.50** | **67.76** |
| Real (fully supervised) | 78.98 | 81.99 |

**DA takes synthetic-only from 25.33 to 79.50 on horse — a +54.2 rescue that
exceeds the fully-supervised baseline.** Domain adaptation is the entire method,
which is the strongest evidence renderer quality is not the binding constraint.

**But note the asymmetry:** tiger only reached 67.76 vs 81.99 supervised. DA
rescued the plain-coated horse fully and the texture-identity tiger partially —
**the same coat-type pattern as zebra/rhino, in an independent dataset. Two
independent replications, both favouring deer.**

**Is a bigger initial gap recoverable?** [UNVERIFIED — no study varies initial gap
size.] Indirect signal: CycleGAN 51.86 / BDL 62.33 / CyCADA 55.57 / CC-SSL 70.77 /
MDAM-MT 79.50 all start from the same Syn 25.33. **The variance is in the
algorithm, not the data quality.**

### ScarceNet, CVPR 2023 — the result that ends the argument [VERIFIED]
https://arxiv.org/abs/2303.15023

**Table 3, PCK@0.05 on TigDog — ScarceNet uses only 36 labelled real images per
category:**

| Method | Horse | Tiger | Average |
|---|---|---|---|
| Real (full supervision, ~8,380 + 6,523 frames) | 78.98 | 81.99 | 80.48 |
| CycleGAN | 51.86 | 46.47 | 49.17 |
| BDL | 62.33 | 52.26 | 57.30 |
| CyCADA | 55.57 | 51.48 | 53.53 |
| CC-SSL | 70.77 | 64.14 | 67.52 |
| MDAM-MT | 79.50 | 67.76 | 73.66 |
| **ScarceNet — 36 real labels/category** | **73.05** | **74.88** | **73.83** |

**36 hand-labelled real images per category matches the entire synthetic +
domain-adaptation pipeline (73.83 vs 73.66) and beats it on tiger by 7.1.**

Authors, verbatim [VERIFIED, §4 L526–528]: *"We also evaluate on this dataset to
show that better performance can be achieved by labeling just a few images, hence
can reduce human labor compared to generating synthetic data."*

Other successors: MAPS (source-free DA, https://arxiv.org/abs/2302.04589),
*A Horse with no Labels* (https://arxiv.org/abs/2308.03411), *Of Mice and Pose*
(https://arxiv.org/abs/2307.13361), DigiDogs (WACV-W 2024 — **GTA V game assets**,
gap bridged with **DINOv2**; another vote for "strong pretrained backbone > better
assets").

---

## 5. The labels-vs-accuracy curve [VERIFIED from ScarceNet]

**AP-10K, mAP, N labelled images *per species* (50 species):**

| Labels/species | Total | HRNet (labels only) | UDA | FixMatch | FlexMatch | **ScarceNet** |
|---|---|---|---|---|---|---|
| 5 | 250 | 0.360 | 0.429 | 0.478 | 0.466 | **0.533** |
| 10 | 500 | 0.463 | 0.519 | 0.544 | 0.555 | **0.597** |
| 15 | 720 | 0.511 | 0.566 | 0.589 | 0.596 | **0.632** |
| 20 | 1000 | 0.547 | 0.580 | 0.601 | 0.618 | **0.654** |
| 25 | 1250 | 0.588 | 0.628 | 0.631 | 0.646 | **0.681** |
| *full (7,023)* | | | | | | *0.738* |

Plain-HRNet is the no-frills baseline: **5 labels → 0.360, 25 → 0.588, full →
0.738.** With semi-supervised refinement, 25 labels/species reaches **0.681 —
92 % of the fully-supervised ceiling.**

Practical curve for a single-species deer project with unlabelled video available:

| Hand-labelled frames | Regime |
|---|---|
| ~10 | Marginal; head keypoints only. Comparable to good synthetic-only. |
| ~50 | **Usable.** Past anything synthetic-only achieves. |
| ~100–200 | **Strong.** ~90 % of ceiling with semi-supervised refinement. **Sweet spot.** |
| ~500+ | Diminishing returns for one species. |

**Effort:** 150 frames × ~20 keypoints ≈ 3,000 clicks. With SuperAnimal zero-shot
pre-placement to *correct* rather than *place*, a single evening. Building,
texturing, rigging, randomising, rendering and then implementing CC-SSL/MDAM-style
DA is **weeks**.

### SuperAnimal-Quadruped as the starting point
Nature Comms 2024 — https://www.nature.com/articles/s41467-024-48792-2

- **39 keypoints**, HRNet-w32, >45 species [REPORTED]
- Quadruped-80K = AwA-Pose, AnimalPose, AcinoSet, Horse-30, StanfordDogs,
  **AP-10K**, iRodent, APT-36K [VERIFIED — https://zenodo.org/records/14016777]
- **Zero-shot on held-out species (Table 2)** [VERIFIED]: AwA-OOD top-down
  **53.4 mAP** vs ImageNet baseline **0**; +pseudo-labelling → 55.4. Openfield
  72.3 → 82.3.
- **Data efficiency:** *"10–100× more data efficient than prior transfer-learning
  approaches."* iRodent 1 % of data → 30–40 mAP vs **0** for ImageNet init.
  [VERIFIED]
- **Honest caveat from the paper:** *"in the limit of enough data, the ImageNet
  methods perform on par with the SuperAnimal models."* The advantage is
  specifically low-label — our regime.
- **Does fine-tuning fix the precision warning?** Model card: not suitable
  zero-shot for high precision, **but** *"can be fine-tuned with minimal data for
  human-level accuracy."* So per the authors, fine-tuning is the documented remedy
  — and the caveat is exactly why the labelling step cannot be skipped. No paper
  reports absolute error in physical units for a new species; treat "human-level"
  as **REPORTED**.

### The one genuine argument FOR synthetic
**Keypoint-set alignment.** Rendering gives labels at exactly our URDF joint
definitions, in 2D **and 3D**, free, at unlimited scale. Hand-labelling gives 2D
only, at a set we must define ourselves, and AP-10K's 17 / SuperAnimal's 39 will
not match our rig.

**Why it still loses:** we can define our own keypoint set when hand-labelling (it
costs nothing extra — we are clicking either way), and 2D keypoints + our URDF +
IK recovers the joint angles. The 3D labels only help if the network is good
enough to use, and synthetic-only networks are not. And the renderer stays
valuable as the IK fitting model regardless.

---

## 6. Deer specifically

**Deer is in AP-10K** [VERIFIED]: Cervidae — **deer 794 labelled images, moose
776**. AP-10K = 10,015 images, 23 families, 54 species, 17 keypoints,
**CC-BY-4.0**. https://arxiv.org/abs/2108.12617

**Deer is transitively inside Quadruped-80K** (AP-10K is a component), so
SuperAnimal has seen deer — **not the unseen-species regime**.

**Inter-family generalisation to Cervidae, trained on Bovidae only** [VERIFIED,
AP-10K Table 9]:

| Test family | mAP |
|---|---|
| Within-Bovidae (reference) | 0.782 ± 0.002 |
| **Cervidae** | **0.641 ± 0.007** |
| Equidae | 0.468 ± 0.019 |
| Hominidae | 0.015 ± 0.001 |

Cervidae retains **82 %** of within-family performance from a model that never saw
a deer. The collapse for Hominidae is driven by *skeletal dissimilarity*, not
appearance.

**ScarceNet's deer result is better still** [VERIFIED, Table 2 — labels from
Bovidae only]:

| Species | Generalisation | Few-Shot (20) | Transfer (all) | **ScarceNet (0 deer labels)** |
|---|---|---|---|---|
| **Deer** | 0.723 | 0.742 | 0.751 | **0.767** |
| Moose | 0.587 | 0.648 | 0.726 | 0.547 |
| Horse | 0.592 | 0.635 | 0.718 | **0.749** |
| Zebra | 0.324 | 0.480 | 0.708 | 0.562 |
| Chimpanzee | 0.009 | 0.022 | 0.550 | 0.057 |

**Deer is the single best-transferring species in the table — 0.767 with zero deer
labels, exceeding training on all deer labels (0.751).** Deer is an easy,
well-covered, structurally-canonical quadruped.

⚠️ **Moose (0.547) is much worse than deer (0.767)** — bulkier silhouette, heavier
antlers. A mild warning that a large or heavily-antlered cervid is harder.

**Antlers:** [UNVERIFIED — no study addresses distinctive appendages.] Closest
signals: moose underperforming deer, and AP-10K's 17-keypoint schema having **no
antler keypoints**, so pretrained models simply ignore them. Practical
implication: antlers are unlabelled clutter that may create false head-keypoint
detections. Argues for including antlered frames in the hand-labelled set, and
mildly for randomising antler geometry if rendering.

---

## 7. Licensing

| Asset | Licence | Risk |
|---|---|---|
| SuperAnimal-Quadruped + Quadruped-80K | **Modified MIT — academic, non-commercial ONLY**; "may not be used to harm any animal"; commercial needs EPFL agreement | **HIGH.** Fine for hobby portfolio if stated; do not vendor weights. [VERIFIED] |
| `JitengMu/Learning-from-Synthetic-Animals` | **GPL-3.0** | **HIGH.** Copying their code makes our repo GPL-3.0. [VERIFIED] |
| TigDog | Non-commercial; creators do not own source video copyright | **HIGH.** Do not redistribute. [REPORTED] |
| AP-10K | CC-BY-4.0 | LOW. [REPORTED] |
| UDA-Animal-Pose | MIT | LOW. [VERIFIED] |
| DeepLabCut core | LGPL-3.0; model-zoo weights carry the Modified MIT above | MEDIUM. Library fine; weights are the constraint. [REPORTED] |
| UE Marketplace / CGTrader / Sketchfab assets | Per-asset EULA; redistribution of source forbidden | MEDIUM — irrelevant if we use our own mesh. |
| **Our own rig + our own labels** | ours | **NONE.** |

**Strategic upside:** the hand-label path with our own rig and labels yields a repo
with **zero third-party data encumbrance** except the pretrained backbone. For a
fully permissive result, fine-tune a **COCO-pretrained HRNet** rather than
SuperAnimal — BEDLAM says COCO-pose pretraining is worth ~4 points PA-MPJPE over
ImageNet and carries no non-commercial clause.

---

## 8. If rendering anyway: prioritised randomisation list

| Rank | Factor | Evidence | Confidence |
|---|---|---|---|
| **1** | **Pose diversity** | SAP identifies limited CAD animation poses as *the* binding failure; CAD+SMAL +1.6 to +2.59 mIoU. PASyn's VAE pose sampler: 30.0→36.7 (+6.7); higher pose variance 91.1→92.4 with real data. Mu et al. had only ~6 canned animations. | **HIGH** |
| **2** | **Background diversity** | Eversberg: random COCO **0.642** vs realistic deployment-domain **0.612** vs HDRI **0.589** — largest single swing (5.3 AP), and randomisation *beats* realism. | **HIGH** |
| **3** | **Camera/viewpoint + scale distribution** | ZebraPose: crop-and-scale augmentation *"significantly improves detection generalization"* — their biggest single lever (mAP50 0.150 → 0.512). | **MED-HIGH** |
| **4** | **Texture randomisation on the animal** | Thalhammer: UV random textures **87.27** vs style transfer 75.88 vs standard 63.22. **Counter:** PASyn says random ImageNet textures widen the gap. **Do both:** ~50 % random, ~50 % plausible deer-brown. | **MED** (conflicting) |
| **5** | **Lighting variation** | Eversberg: white 0.623 → temperature-varied 0.635 → HDRI IBL **0.642**. Modest (+1.9 AP), cheap in Blender, impossible in matplotlib. | **MED** |
| **6** | **Distractors / occluders** | Eversberg: none 0.653 → YCB 0.660 → **random-textured cubes 0.669**. Cheap primitives beat realistic objects. Real footage has branches and grass. | **MED** |
| **7** | **Surface realism (fur, SSS, PBR)** | Eversberg: grey **0.642** vs best realistic material **0.653** — **1.1 AP across the entire realism range**, smallest lever measured. SPAC-Net: rhino **+0.0** once 99 real labels existed. | **LOW — deprioritise** |

**Renderer:** use **Blender, not matplotlib/trimesh**. Ranks 2, 5 and 6 are trivial
in Blender and awkward-to-impossible in matplotlib; rank 4 needs real UV handling.
CPU-only Cycles is fine — PASyn and SPAC-Net both got their results with **3,000
Blender-rendered images**, an overnight CPU job.

### The high-leverage move if synthetic scale is ever wanted
**Generative appearance upgrade.** Render depth maps + Canny edges + ground-truth
keypoints from the crude rig, feed as ControlNet/FLUX conditioning. The mesh only
has to be geometrically right — **surface appearance comes from the diffusion
model.**
- SPAC-Net: synthetic-only zebra 46.6 → **80.4** this way. [VERIFIED]
- Generative Zoo (https://arxiv.org/abs/2412.08101): FLUX on SMAL depth+Canny, 1M
  images, trained **exclusively synthetic**, S-MPJPE **160.1** vs best baseline
  **374.9**; PCK@0.5 **97.0 %** vs 85.6 % on real Animal3D. Ablation: depth-only =
  most realistic but poorest pose alignment; Canny-only = best alignment, worse
  realism; **combined = best**. [REPORTED]

**This is the field's actual answer to "my renderer is crude" — a far better use
of effort than mesh work.**

---

## 9. Unknowns and the cheap experiment for each

| # | Unknown | Status | Cheapest decisive experiment |
|---|---|---|---|
| 1 | Does mesh/texture fidelity affect **animal** 2D keypoint PCK, holding poses/camera/background fixed? | **GENUINELY UNMEASURED** | Render 3 conditions from our own rig — flat grey / random UV textures / best deer texture — 2,000 images each, identical poses+cameras+backgrounds. Train identical HRNets. Evaluate on 100 hand-labelled real frames. **A publishable-grade ablation nobody has run.** |
| 2 | How many hand-labelled deer frames for usable accuracy on *our* footage and keypoint set? | Extrapolated, not measured | Label 200 once; train on 10/25/50/100/200 subsets; evaluate on held-out 50. One afternoon. **Do this first.** |
| 3 | Does SuperAnimal zero-shot suffice to bootstrap labelling? | Model card silent on bootstrap value | Run zero-shot on 20 frames; time correcting vs labelling from scratch. 30 min. |
| 4 | Do antlers cause head-keypoint failures? | **UNMEASURED** | SuperAnimal zero-shot on 20 antlered vs 20 antlerless; compare head-keypoint error. 20 min. |
| 5 | Is the "deer is rhino-like, not zebra-like" analogy valid? | Inferred from two independent replications, never tested on deer | Falls out of experiment #1 free. |
| 6 | Does DA rescue *our* renderer's gap specifically? | Unknowable a priori | Only worth answering if #2 shows hand-labelling infeasible. |
| 7 | BEDLAM's clothing-realism magnitude? | **Numbers do not exist in the paper** | Not settleable from published material. |
| 8 | Does ControlNet appearance-upgrade work with a *hobby* mesh? | Untested (SPAC-Net used CGTrader, Generative Zoo used SMAL) | Render 200 depth+Canny pairs, run through a ControlNet checkpoint, eyeball whether anatomy survives. One evening. |

---

## 10. Sources

**Primary, full PDF text — VERIFIED:**
- Mu et al., *Learning from Synthetic Animals*, CVPR 2020 — https://arxiv.org/abs/1912.08265 · https://github.com/JitengMu/Learning-from-Synthetic-Animals
- Li & Lee, *From Synthetic to Real: UDA for Animal Pose*, CVPR 2021 — https://arxiv.org/abs/2103.14843 · https://github.com/chaneyddtt/UDA-Animal-Pose
- Li & Lee, *ScarceNet*, CVPR 2023 — https://arxiv.org/abs/2303.15023
- Jiang & Ostadabbas, *PASyn*, BMVC 2022 — https://arxiv.org/abs/2208.13944
- Jiang & Ostadabbas, *SPAC-Net*, 2023 — https://arxiv.org/abs/2305.17845
- Ye et al., *SuperAnimal*, Nature Comms 2024 — https://www.nature.com/articles/s41467-024-48792-2
- Thalhammer et al., *Shape-biased Texture Agnostic Representations*, 2024 — https://arxiv.org/abs/2402.04878

**Primary, HTML conversion — VERIFIED with that caveat:**
- Yu et al., *AP-10K*, NeurIPS D&B 2021 — https://arxiv.org/abs/2108.12617
- Black et al., *BEDLAM*, CVPR 2023 — https://arxiv.org/abs/2306.16940
- Peng et al., *Learning Part Segmentation from Synthetic Animals*, WACV-W 2024 — https://arxiv.org/abs/2311.18661
- Eversberg & Lambrecht, Sensors 2021 — https://www.mdpi.com/1424-8220/21/23/7901

**Secondary — REPORTED:**
- Tobin et al., *Domain Randomization*, IROS 2017 — https://arxiv.org/abs/1703.06907
- Geirhos et al., ICLR 2019 — https://arxiv.org/abs/1811.12231
- Hodaň et al., ICIP 2019 — https://arxiv.org/abs/1902.03334
- Sengupta et al., *STRAPS*, BMVC 2020 — https://arxiv.org/abs/2009.10013
- *Generative Zoo*, 2024 — https://arxiv.org/abs/2412.08101
- *ZebraPose*, 2024 — https://arxiv.org/abs/2408.10831
- *DigiDogs*, WACV-W 2024 — https://cvssp.org/data/DigiDogs/
- SuperAnimal model card — https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped · https://zenodo.org/records/14016777
- Animal Pack Ultra specs — https://assetsue.com/file/animal-pack-ultra
- TigDog — https://calvin-vision.net/datasets/tigdog/
