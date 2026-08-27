# Historical Phase-B training code

This directory implements the zero-shot/enrollment evidence-engine experiments that preceded the
application pivot. It includes memory construction, episodic candidate-label training, retrieval,
reranking, and the corresponding legacy evaluators.

It is retained for reproducibility and branch interoperability. It is **not** the design of record
for arbitrary movement detection, movement-difference quantification, or recurrent motif discovery.
New application code must not import candidate sets, label aliases, evidence voting, or Phase-B
episode assumptions from this directory.

The historical documentation and exact results are available at commit `32267b6`. The active
application architecture and staged replacement are documented in:

- [`../../docs/design/DESIGN_OF_RECORD.md`](../../docs/design/DESIGN_OF_RECORD.md)
- [`../../docs/design/EVALUATION_PROTOCOL.md`](../../docs/design/EVALUATION_PROTOCOL.md)
- [`../../docs/design/IMPLEMENTATION_PLAN.md`](../../docs/design/IMPLEMENTATION_PLAN.md)
