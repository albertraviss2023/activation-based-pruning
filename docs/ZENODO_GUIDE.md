# Zenodo DOI Integration Guide (Option 2)

To complete the academic release and obtain a DOI for your thesis, follow these steps:

## 1. Finalize the Repository
1. Ensure all changes (LICENSE, CITATION.cff, README.md updates) are committed and pushed to GitHub.
2. Clear the outputs of your notebooks if you want a cleaner repo (optional but recommended).
3. Ensure `.gitignore` is correctly excluding large data files and checkpoints.

## 2. Connect GitHub to Zenodo
1. Log in to [Zenodo.org](https://zenodo.org/) using your GitHub account.
2. Navigate to the "GitHub" section in your Zenodo profile.
3. Find the `activation-based-pruning` repository and flip the switch to **On**.

## 3. Create a Release
1. Go to your GitHub repository.
2. Click on **Releases** -> **Create a new release**.
3. Tag the version (e.g., `v0.88.0`).
4. Give it a title and description.
5. Click **Publish release**.

## 4. Retrieve the DOI
1. Once the release is published, Zenodo will automatically archive the repository and mint a DOI.
2. Go back to Zenodo -> "Uploads" to find the new entry.
3. Copy the DOI (it will look like `10.5281/zenodo.XXXXXXX`).

## 5. Update Metadata (The "DOI Loop")
1. In your local repository, update the `FIXME` placeholders in `README.md` and `CITATION.cff` with the actual DOI.
2. Commit and push these changes.
3. (Optional) Create a second release (e.g., `v0.88.1`) if you want the DOI to point to a version that actually contains the DOI in the text. Zenodo allows you to update metadata later, but a new release is cleaner for a "fixed" record.

## 6. Cite in your Thesis
Use the BibTeX provided in the README or the citation suggested by Zenodo.
