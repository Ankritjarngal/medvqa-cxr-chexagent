# Disease naming, first run (text-only: the image never reached the model)

Notebook 04 as first built, run on Colab 10 Nov 2026. A bug in how transformers 4.46.3 and the model's own code interact
meant the image was never encoded (`docs/experiment_log.md`, entry 17). The model saw the text of the prompt only, including
the image file path as plain text.

What the data shows, and why it matches that cause:
- `open`: the answer was "Pneumonia" for all 130 images. `forced`: "No Finding" for all 130. `perclass`: the highest score was
  "Infiltration" for all 130.
- The per-class scores vary between images only in the second or third decimal (standard deviation 0.002 to 0.009), which is
  what different file-path text would produce.

Keep as a text-only control: it shows what this model answers when it cannot see the image. It is **not** a measure of whether
CheXagent can name a disease from an X-ray. Rerun with the fixed notebook (`notebooks/04_disease_naming.ipynb`).
