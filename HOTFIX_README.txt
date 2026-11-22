FedTwin-RUL final-study recursion hotfix

Fixes:
1. FedTwin official-test inference accidentally called predict_dispatch()
   recursively. It now correctly calls predict_tcm().
2. Validation prediction now consistently routes through predict_dispatch().

Your already completed result.json runs are safe.

Extract this ZIP over the project root and replace the existing script.

Resume the SAME study with PowerShell:

$study = Get-Content outputs\latest_fedtwin_final.txt
python scripts\run_fedtwin_final_study.py --config configs\fedtwin_final.yaml --resume --study-dir "$study"

Do not run the plain full command again, because that would create a new study directory.
