# Colab and GitHub workflow

## One-time setup

1. **Create the GitHub repo** as private, named `fedcti-llm-v3`, with no README (this repo
   already has one). Then push from your machine:

   ```bash
   git remote add origin https://github.com/khalerishabh/fedcti-llm-v3.git
   git push -u origin main
   ```

2. **Create a fine-grained token** at GitHub > Settings > Developer settings > Personal access
   tokens > Fine-grained tokens:
   - Repository access: only `fedcti-llm-v3`
   - Permissions: Contents = Read and write (Metadata = Read is added automatically)
   - Expiry: e.g. 90 days

3. **Store it in Colab**: open any notebook, click the key icon in the left sidebar, add a
   secret named `GITHUB_TOKEN`, paste the token, and enable "Notebook access".
   Never paste the token into a cell.

4. **Let Colab open private notebooks**: File > Open notebook > GitHub tab, tick
   "Include private repos", and authorize when GitHub asks.

5. `GITHUB_USER` in `notebooks/00_colab_setup.ipynb` is already set to `khalerishabh`.

## Every session

1. Open `00_colab_setup.ipynb` from GitHub and run all cells. It mounts Drive, clones or
   pulls the repo into `/content/fedcti-llm-v3`, and installs it.
2. Open the experiment notebook (File > Open notebook > GitHub) and run it. Notebooks start
   with a short cell that puts `/content/fedcti-llm-v3/src` on `sys.path`.
3. Results land on Drive under `fedcti_v3/`. Stages already completed are skipped.

## Rules

- Code changes are made in the repo, not in Colab. If you edit a notebook in Colab, save it
  with File > Save a copy in GitHub, or download it into the repo and commit.
- Commits use your identity: `Rishabh Khale <khalerishabhproff@gmail.com>`. The setup
  notebook sets the same identity inside Colab.
- Commit notebooks without outputs (`nbstripout --install` once locally).
- Never commit data, adapters, tokens or `.env` files; `.gitignore` covers these.
