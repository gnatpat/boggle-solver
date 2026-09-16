# Deployment

The web app in `static/` is deployed with [npd](https://github.com/gnatpat/npd)
as a static site — no server process.

- `npd.toml` at the repo root is the whole deploy config: which directory to
  publish and the route it is served under.
- **Pushing to `main` deploys it.** `.github/workflows/deploy.yml` just asks the
  server to pull this repo and republish `static/`; nothing is uploaded by CI.
- `static/boggle_cnn.onnx` and `static/tries/` are generated offline (they need
  the training images and the SCOWL word lists, which are not in git) and are
  committed as built artifacts. A clone must be deployable as-is, so if you
  regenerate them, commit them.
- `npd dev` serves the site locally.
- Please do not add a second deploy mechanism (Dockerfile, deploy script, CI
  deploy step) without asking.
