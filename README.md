# Learning Peptide Properties with Positive Examples Only

https://github.com/ur-whitelab/pu-peptides/assets/51170839/ff24ecfd-7782-4f95-b964-4be2aa09c16e

Deep learning can create accurate predictive models by exploiting existing large-scale experimental data, and guide the design of molecules. However, a major barrier is the requirement of both positive and negative examples in the classical supervised learning frameworks. Notably, most peptide databases come with missing information and low number of observations on negative examples, as such sequences are hard to obtain using high-throughput screening methods. To address this challenge, we solely exploit the limited known positive examples in a semi-supervised setting, and discover peptide sequences that are likely to map to certain antimicrobial properties via positive-unlabeled learning (PU). In particular, we use the two learning strategies of adapting base classifier and reliable negative identification to build deep learning models for inferring solubility, hemolysis, binding against SHP-2, and non-fouling activity of peptides, given their sequence. We evaluate the predictive performance of our PU learning method and show that by only using the positive data, it can achieve competitive performance when compared with the classical positive-negative (PN) classification approach, where there is access to both positive and negative examples.

![toc](https://github.com/ur-whitelab/pu-peptides/assets/51170839/ed5308a5-614f-42a8-a3bf-2a903506d4a9)

## Citation

See [paper](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d3dd00218g) and the citation:

```bibtex
@article{ansari2024learning,
  title={Learning peptide properties with positive examples only},
  author={Ansari, Mehrad and White, Andrew D},
  journal={Digital Discovery},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```
