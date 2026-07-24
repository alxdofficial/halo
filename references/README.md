# References — datasets & baselines

Local copy of the canonical **paper** (or landing page) + a machine-readable
`citation.json` (title, authors, year, venue, DOI, URLs, license, BibTeX) for every
dataset and baseline HALO uses. Purpose: (1) easy citation, (2) a code-vs-paper
cross-checking run (are baselines represented correctly? datasets used per their
documented protocol? channel descriptions accurate?).

Each `references/{datasets,baselines}/<name>/` holds:
`paper.pdf` **or** `webpage.html`, `citation.json`, `SOURCE.txt` (data/code + paper URLs).

**Status: 22 full PDFs, 4 landing pages (paywalled / no open version), 0 missing. All 26 `citation.json` present + valid.**

## Datasets (19)

| name | role | paper | year | venue | local | data/code source | license |
|---|---|---|---:|---|---|---|---|
| **capture24** | train | [CAPTURE-24: A large dataset of wrist-worn activity tracker data collec](https://www.nature.com/articles/s41597-024-03960-3) | 2024 | Scientific Data, vol. 11, Article no. 1135 ( | `paper.pdf` | https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001 | CC BY 4.0 |
| **dsads** | train | [Recognizing Daily and Sports Activities in Two Open Source Machine Lea](https://academic.oup.com/comjnl/article-abstract/57/11/1649/411286) | 2014 | The Computer Journal, Vol. 57, No. 11, pp. 1 | `paper.pdf` | https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities | Dataset (UCI): CC BY 4.0. Paper: (c) The |
| **extrasensory** | planned-test | [Recognizing Detailed Human Context In-the-Wild from Smartphones and Sm](https://arxiv.org/abs/1609.06354) | 2017 | IEEE Pervasive Computing, vol. 16, no. 4, Oc | `paper.pdf` | http://extrasensory.ucsd.edu/ | Free for research use; must cite the pap |
| **hapt** | train | [Transition-Aware Human Activity Recognition Using Smartphones](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930) | 2016 | Neurocomputing, vol. 171, pp. 754-767 (Elsev | `webpage.html` | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions | CC BY 4.0 (dataset, per UCI ML Repositor |
| **harth** | test | [HARTH: A Human Activity Recognition Dataset for Machine Learning](https://www.mdpi.com/1424-8220/21/23/7853) | 2021 | Sensors (MDPI), 21(23):7853 | `paper.pdf` | https://archive.ics.uci.edu/dataset/779/harth | CC BY 4.0 |
| **hhar** | train | [Smart Devices are Different: Assessing and Mitigating Mobile Sensing H](https://dl.acm.org/doi/10.1145/2809695.2809718) | 2015 | Proceedings of the 13th ACM Conference on Em | `paper.pdf` | https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition | Dataset: CC BY 4.0 (UCI ML Repository, D |
| **inclusivehar** | test | [InclusiveHAR: A Smartphone-Based Dataset for Human Activity Recognitio](https://data.mendeley.com/datasets/r78dn3f6nc/4) | 2026 | Mendeley Data (v4) | `webpage.html` | https://data.mendeley.com/datasets/r78dn3f6nc/4 | CC BY 4.0 |
| **kuhar** | train | [KU-HAR: An open dataset for heterogeneous human activity recognition](https://www.sciencedirect.com/science/article/abs/pii/S0167865521000933) | 2021 | Pattern Recognition Letters, vol. 146, pp. 4 | `webpage.html` | https://data.mendeley.com/datasets/45f952y38r/5 | Dataset (Mendeley Data): CC BY 4.0; code |
| **mhealth** | train | [Design, implementation and validation of a novel open framework for ag](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-14-S2-S6) | 2015 | BioMedical Engineering OnLine, 14(Suppl 2):S | `paper.pdf` | https://archive.ics.uci.edu/dataset/319/mhealth+dataset | CC BY 4.0 (dataset, UCI); journal paper  |
| **mobiact** | test | [The MobiAct Dataset: Recognition of Activities of Daily Living using S](https://www.scitepress.org/papers/2016/57924/) | 2016 | Proceedings of the 2nd International Confere | `paper.pdf` | https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/ | Non-commercial research/educational use  |
| **motionsense** | test | [Mobile Sensor Data Anonymization](https://arxiv.org/abs/1810.11546) | 2019 | Proceedings of the International Conference  | `paper.pdf` | https://github.com/mmalekzadeh/motion-sense | MIT (code/data repository) |
| **opportunity** | test | [Collecting complex activity datasets in highly rich networked sensor e](https://doi.org/10.1109/INSS.2010.5573462) | 2010 | 2010 Seventh International Conference on Net | `paper.pdf` | https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition | CC BY 4.0 (dataset, per UCI). Conference |
| **pamap2** | train | [Introducing a New Benchmarked Dataset for Activity Monitoring](https://doi.org/10.1109/ISWC.2012.13) | 2012 | 16th International Symposium on Wearable Com | `paper.pdf` | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | Dataset (UCI): CC BY 4.0. Canonical pape |
| **realworld** | test | [On-body Localization of Wearable Devices: An Investigation of Position](https://ieeexplore.ieee.org/document/7456521) | 2016 | 2016 IEEE International Conference on Pervas | `paper.pdf` | https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/ | unknown |
| **recgym** | train | [The Contribution of Human Body Capacitance/Body-Area Electric Field To](https://arxiv.org/abs/2210.14794) | 2022 | arXiv preprint arXiv:2210.14794 (dataset lat | `paper.pdf` | https://archive.ics.uci.edu/dataset/1128 | CC BY 4.0 (UCI dataset) |
| **shoaib** | test | [Fusion of Smartphone Motion Sensors for Physical Activity Recognition](https://www.mdpi.com/1424-8220/14/6/10146) | 2014 | Sensors (MDPI), 14(6):10146-10176 | `paper.pdf` | https://www.utwente.nl/en/eemcs/ps/research/dataset/ | CC BY 4.0 (paper, MDPI open access); dat |
| **uci_har** | train | [A Public Domain Dataset for Human Activity Recognition Using Smartphon](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf) | 2013 | ESANN 2013 - 21st European Symposium on Arti | `paper.pdf` | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones | Dataset: CC BY 4.0 (UCI Machine Learning |
| **unimib_shar** | train | [UniMiB SHAR: A Dataset for Human Activity Recognition Using Accelerati](https://www.mdpi.com/2076-3417/7/10/1101) | 2017 | Applied Sciences (MDPI), Vol. 7, Issue 10, A | `paper.pdf` | http://www.sal.disco.unimib.it/technologies/unimib-shar/ | CC BY 4.0 (article, per Crossref/MDPI);  |
| **wisdm** | train | [Smartphone and Smartwatch-Based Biometrics Using Activities of Daily L](https://ieeexplore.ieee.org/document/8835065) | 2019 | IEEE Access, vol. 7, pp. 133190-133202 | `paper.pdf` | https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset | Paper: CC BY (IEEE Access, gold open acc |

## Baselines (7)

| name | role | paper | year | venue | local | data/code source | license |
|---|---|---|---:|---|---|---|---|
| **crosshar** | active | [CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hi](https://doi.org/10.1145/3659597) | 2024 | Proceedings of the ACM on Interactive, Mobil | `webpage.html` | https://github.com/kingdomrush2/CrossHAR | unknown (code repo has no LICENSE file;  |
| **lanhar** | dropped (related-work) | [Large Language Model-Guided Semantic Alignment for Human Activity Reco](https://arxiv.org/abs/2410.00003) | 2024 | arXiv preprint (cs.CV); latest v4 Oct 2025 | `paper.pdf` | https://github.com/DASHLab/LanHAR | arXiv.org perpetual non-exclusive licens |
| **limubert** | active | [LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing ](https://dl.acm.org/doi/10.1145/3485730.3485937) | 2021 | SenSys '21: Proceedings of the 19th ACM Conf | `paper.pdf` | https://github.com/dapowan/LIMU-BERT-Public | MIT (code repo); paper (c) ACM |
| **llasa** | dropped (related-work) | [LLaSA: A Sensor-Aware LLM for Natural Language Reasoning of Human Acti](https://arxiv.org/abs/2406.14498) | 2024 | arXiv preprint arXiv:2406.14498 (cs.CL); als | `paper.pdf` | https://github.com/BASHLab/LLaSA | Project website CC BY-SA 4.0; code repo  |
| **moment** | dropped (related-work) | [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885) | 2024 | International Conference on Machine Learning | `paper.pdf` | https://github.com/moment-timeseries-foundation-model/moment | Code: MIT (repo). Paper: arXiv CC BY 4.0 |
| **ssl-wearables** | planned | [Self-supervised learning for human activity recognition using 700,000 ](https://www.nature.com/articles/s41746-024-01062-3) | 2024 | npj Digital Medicine, 7:91 (2024). Preprint: | `paper.pdf` | https://github.com/OxWearables/ssl-wearables | Custom academic/non-commercial license ( |
| **unimts** | planned | [UniMTS: Unified Pre-training for Motion Time Series](https://arxiv.org/abs/2410.19818) | 2024 | Advances in Neural Information Processing Sy | `paper.pdf` | https://github.com/xiyuanzh/UniMTS | Paper: CC BY 4.0 (arXiv). Code repo: no  |

## Landing-page-only (no open PDF — fetch manually if the paper text is needed)

- **hapt** — Neurocomputing (Elsevier, paywalled). Protocol documented on the UCI page.
- **kuhar** — Pattern Recognition Letters (Elsevier, paywalled). Dataset on Mendeley (CC BY).
- **crosshar** — IMWUT/ACM + OpenReview (login-gated PDF). Method also in the vendored `auxiliary_repos/CrossHAR` code, so code-level cross-check is possible without the PDF.
- **inclusivehar** — Mendeley **data-only deposit; no associated paper exists** (cite the dataset DOI 10.17632/r78dn3f6nc.4). The "Mobility Disabilities" Scientific Data paper is a *different* dataset.

_Generated from the per-item `citation.json` files. Regenerate after adding an item._
