# Upstream references and licenses

O1FM is an independent implementation. The initial architecture study used:

- Google Research, `google-research/timesfm`, commit
  `7360853c4f8ea28bb1b3eaf5b7af2d8e6b8fcf05` (2026-09-02 checkout).
- Google Research, “TimesFM-3: A zero-shot foundation model for multivariate
  forecasting,” published 2026-08-31.
- Hugging Face model card `google/timesfm-3.0-pytorch`.

The official source repository is Apache-2.0. The TimesFM-3 pretrained weights
are currently released under `timesfm-non-commercial-license-v1.0` and are not
vendored, downloaded, or redistributed by O1FM.

O1FM's implementation, state formats, O1 stream, graph, Forecast Crystals, and
Action Bank are original project code. Architectural facts are attributed in
the public README and documentation.
