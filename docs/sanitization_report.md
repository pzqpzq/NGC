# Sanitization Report

## Actions Taken

- Copied only selected scripts, source-data tables, representative figures, and a reduced candidate-bank sample.
- Excluded model loaders with private local model/dataset paths.
- Excluded full logs, remote inventories, raw prediction dumps, checkpoints, and cache directories.
- Replaced private absolute path defaults in copied scripts with environment-variable or repo-relative defaults.
- Reduced the sample candidate bank to six manuscript-relevant model/dataset pairs and removed prompt/answer text columns.

## Local Scan

Run from the package root before publishing:

```bash
rg -n --hidden 'ict[0-9]+|[s]ervernew|/data/[p]eizhengqi|/home/[p]eizhengqi|p[a]ssword|p[a]sswd|a[p]i[_-]?k[e]y|s[e]cret|s[k]-[A-Za-z0-9_-]{20,}|[O]PENAI|[A]NTHROPIC|[S]ILICONFLOW' .
```

The scan was run after curation and returned no matches for the copied release tree.

## Remaining Review Items

- If new live-inference files are added later, inspect them manually for model paths, dataset paths, credentials, and hostnames.
- If full prediction outputs are added later, consider whether benchmark prompts/answers can be redistributed.
- If the package is pushed to GitHub, use GitHub credential scanning or an equivalent tool as a second check.
