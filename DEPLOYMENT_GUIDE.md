# ResolveAI v6 — Cloud candidate (not yet deployed)

This folder is a **small deployment candidate**, derived from the tested v6 source. The original local application has not been modified. All banking policies are **fictional demonstration documents**. Do not enter personal financial data or credentials.

## Important limitations

- The full BANKING77 dataset is **not included**. With no `data/banking77.json`, the app uses the 60-example demonstration classifier, which is not equivalent to the reported full BANKING77 results. This fallback is appropriate for a functional smoke demo, **not** to claim BANKING77 accuracy.
- Public Streamlit Community Cloud does **not** have access to Ollama on your Mac. Use the default **Template** response mode for cloud testing. OpenAI is optional and requires secret configuration and careful privacy assessment.
- Semantic retrieval is optional and not installed by `requirements.txt`. The default is TF-IDF.
- The 52-case routing agreement of 38/52 and 24-case retrieval Hit@1 of 15/24 are development-authored measures, not independent production performance estimates.
- No live hosted app or full Streamlit runtime was verified in the preparation environment. Public deployment should be validated separately before sharing it with a professor.

## Local Mac run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_check.py
python -m streamlit run app.py
```

To reproduce the full classifier locally, copy your own validated `banking77.json` into `data/` before restarting the app, and select BANKING77 in Developer Settings. Do not publish the dataset to a public GitHub repository without checking its distribution terms and whether your repo should be public.

## Streamlit Community Cloud

1. Create a **new GitHub repository** and upload only the files in this folder; **do not** upload `.venv`, credentials, account data, or historic backups.
2. Go to https://share.streamlit.io and use **Create app** → **Yup, I have an app**.
3. Select your GitHub repository and branch; set the entrypoint to `app.py`.
4. In **Advanced settings**, choose Python **3.12** if offered (rather than copying the local Python 3.9 environment). Add secrets only if enabling an external provider.
5. Deploy, then check OOD rejection, a supported card query, a bank transfer, and credential-disclosure escalation using fictional inputs. Use **Template** response mode and leave Semantic RAG off initially.
6. Verify the UI loads and the browser shows a real `streamlit.app` URL before claiming public deployment.

## Evaluation

`python smoke_check.py` verifies importable core behavior and one retrieval case. For an official BANKING77 evaluation, provide the complete dataset and install `matplotlib`, then run `python evaluate_ml.py` from the folder. The script writes to `evaluation_results/`; neither the dataset nor results are included here.

## Release gate

- [ ] Hosted app launches successfully
- [ ] Cloud default indicates small demo classifier (or full BANKING77 is intentionally provided and validated)
- [ ] 4 smoke scenarios pass through the deployed UI
- [ ] No real credentials or sensitive client data entered
- [ ] Release limitations and human-review requirement visible
