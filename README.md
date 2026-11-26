# Sai Surya Varun Bade

### Evidence, uncertainty, and human judgment in analytical systems

I build analytical software and explore a question it cannot answer on its own: **what evidence should a system provide before someone acts on its output?** My interests connect information systems, reliable machine learning, and human-AI decision support.

I completed an MS in Information Systems with a Business Analytics concentration at California State University, Fullerton. My background includes Python/SQL workflows, dashboards, and backend implementation. My master's was non-thesis; the work below is independent and exploratory, not a list of peer-reviewed publications.

## Research portfolio

| Work | Question | Evidence and current status |
|---|---|---|
| [Selective prediction under changing missingness](research/missingness/) | Does data-quality information improve referral decisions beyond confidence when labels are limited? | Existing exploratory experiments across four datasets; released aggregates, source snapshot, and runnable arithmetic audit. Mixed and negative results. |
| [Evidence review for financial explanations](research/evidence-review/) | What can linked records verify, and when should a reviewer defer? | New deterministic synthetic diagnostic, reproducible fixtures, tests, and a proposed human-study protocol. No human or LLM evaluation. |
| [VarianceIQ](https://github.com/saisuryavarun-bade/varianceiq) | How can software make financial investigation steps inspectable? | An agentic software prototype using analytical tools and streamed outputs. Its existence does not establish improved human decisions. |

## Start here

1. Read the missingness study's [result and limitations](research/missingness/README.md).
2. Run its aggregate audit and the [evidence-review diagnostic](research/evidence-review/README.md).
3. Examine the [proposed experiment](research/evidence-review/STUDY_PROTOCOL.md): evidence access crossed with review accountability.

```bash
git clone https://github.com/saisuryavarun-bade/SaiSuryaVarun-Bade.git
cd SaiSuryaVarun-Bade
python research/missingness/audit_aggregates.py
python research/evidence-review/benchmark.py
python -m unittest discover -s research/evidence-review -p "test_*.py" -v
```

These entry-point commands require Python 3.11+ and only its standard library. Full model-training replication of the archived missingness study is a separate, not-yet-verified release step.

## Earlier implementation projects

- [Chronic disease modeling](https://github.com/saisuryavarun-bade/ml-disease-predictor): a modeling exercise with synthetic, rule-labeled records. Reported accuracy is not clinical validation.
- [Healthcare financial analytics](https://github.com/saisuryavarun-bade/data-analyst-health-finance): analytics and dashboard implementation.
- [MedFinBot](https://github.com/saisuryavarun-bade/ai-chatbot-finhealth): a retrieval-based chatbot prototype. Not clinical or financial advice.

## How I present the work

I distinguish implementations from empirical evidence, keep adverse results visible, and identify limits on reproducibility. New portfolio code and writing were developed with AI assistance; the new artifacts are undergoing author review. No study participants, institutional affiliation, faculty collaboration, or publication is implied.

[LinkedIn](https://www.linkedin.com/in/saisuryavarun-bade/) · [Email](mailto:saisuryavarun.bade@gmail.com)
