# Research Through Evaluation (RtE) for Large Language Models in Healthcare

**Authors:** Yuexing Hao, Jason Holmes, Jared Hobson, Alexandra Bennett, Elizabeth L. McKone, Daniel K. Ebner, David M. Routman, Satomi Shiraishi, Samir H. Patel, Nathan Y. Yu, Chris L. Hallemeier, Brooke E. Ball, Saleh Kalantari, Marzyeh Ghassemi, Mark Waddle, Wei Liu

**Affiliations:** Mayo Clinic · Massachusetts Institute of Technology · Cornell University

**Corresponding authors:** Yuexing Hao (yuexing@mit.edu) · Wei Liu (liu.wei@mayo.edu)

---

Research through Evaluation (RtE) is an iterative framework that engages clinical professionals to collaboratively refine evaluation metrics through small-scale exploratory studies. This repository contains the dataset, analysis code, and figures associated with the paper.

🌐 **Project website:** [yuexinghao.github.io/Research_Through_Evaluation](https://yuexinghao.github.io/Research_Through_Evaluation)

---

## Repository Structure

```
Research_Through_Evaluation/
│
├── Data/
│   └── In-Basket-QA-Dataset - Public.csv   # Patient in-basket messages with
│                                            # In-Basket Bot and clinician responses,
│                                            # type labels, and category annotations
│
├── Code/
│   ├── category_analysis.py                 # Per-category performance analysis:
│   │                                        # radar chart, bar chart, donut chart,
│   │                                        # and grader comment word cloud
│   │
│   └── clinician_grading_analysis.py        # Clinician grading analysis:
│                                            # descriptive stats, ICC, bias plots,
│                                            # multi-round bar chart, LLM grader
│                                            # scatter plot, editing requirement charts
│
├── Fig/
│   ├── RtE_ Research through Evaluation.png # RtE framework overview
│   ├── RtE_Development.png                  # Development pipeline
│   ├── LLM_Graders_Radar.png               # LLM grader radar chart
│   ├── Clinician Graders.png               # Clinician grader results
│   ├── Three_Clinician_Results.png         # Three-clinician comparison
│   ├── Readability_Combination.png         # Readability metrics
│   └── Logo/                               # Institutional logos (Mayo, MIT, Cornell)
│
├── index.html                               # Project website — main page (figures)
├── data.html                                # Project website — dataset viewer
├── LICENSE
└── README.md
```

## Dataset

`Data/In-Basket-QA-Dataset - Public.csv` contains radiation oncology patient in-basket messages paired with responses from both an LLM-based In-Basket Bot and clinicians. Columns:

| Column | Description |
|--------|-------------|
| `MessageNote` | De-identified patient in-basket message |
| `Response.1` | Response to the message |
| `Types` | Responder type: `In-Basket Bot` or `Clinicians` |
| `Category` | Clinical topic category (e.g. *Side Effects*, *Medication Question*) |

Each patient message typically appears twice — once with the In-Basket Bot response and once with the clinician response.

## Code

Both scripts are self-contained and run locally (no Colab required).

**Dependencies:**
```bash
pip install pandas openpyxl matplotlib seaborn scipy scikit-learn pingouin statsmodels wordcloud nltk
python -m nltk.downloader punkt averaged_perceptron_tagger
```

**Running the scripts:**
```bash
# Category-level performance analysis
python Code/category_analysis.py --data-dir Data --out-dir Code/output

# Clinician grading analysis
python Code/clinician_grading_analysis.py --data-dir Data --out-dir Code/output

# Skip slow ICC computation during development
python Code/clinician_grading_analysis.py --skip-icc
```

Output figures are saved to `Code/output/` by default.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
