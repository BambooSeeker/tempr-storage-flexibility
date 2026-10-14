# Release maintenance

Repository: https://github.com/BambooSeeker/tempr-storage-flexibility
Version: 1.2.0

1. Distribute only the public release files listed in manifest.csv.
2. Preserve the MIT license, market data terms and third-party notices.
3. Run `python reproduce.py --check` and `python -m unittest discover -s tests -v`.
4. Inspect the GitHub Actions result for each published commit. The optional live-inference workflow is manually triggered.
5. Regenerate manifest.csv after changing any tracked file and create a version tag for each release.
6. Update CITATION.cff with the final manuscript DOI when assigned.

The public package includes the original LoRA adapter and limited example inputs. Full market
records, internal audits, credentials, environments and pretrained base weights are excluded.
