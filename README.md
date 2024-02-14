# Learning Peptide Properties with Positive Examples Only



https://github.com/ur-whitelab/pu-peptides/assets/51170839/8243ae83-adc9-4663-8455-5d8d58c784ed


Deep learning can create accurate predictive models by exploiting existing large-scale experimental data, and guide the design of molecules. However, a major barrier is the requirement of both positive and negative examples in the classical supervised learning frameworks. Notably, most peptide databases come with missing information and low number of observations on negative examples, as such sequences are hard to obtain using high-throughput screening methods. To address this challenge, we solely exploit the limited known positive examples in a semi-supervised setting, and discover peptide sequences that are likely to map to certain antimicrobial properties via positive-unlabeled learning (PU). In particular, we use the two learning strategies of adapting base classifier and reliable negative identification to build deep learning models for inferring solubility, hemolysis, binding against SHP-2, and non-fouling activity of peptides, given their sequence. We evaluate the predictive performance of our PU learning method and show that by only using the positive data, it can achieve competitive performance when compared with the classical positive-negative (PN) classification approach, where there is access to both positive and negative examples.

![toc](https://github.com/ur-whitelab/pu-peptides/assets/51170839/ed5308a5-614f-42a8-a3bf-2a903506d4a9)

## Citation

See [pre-print](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01317) and the citation:

```bibtex
@article{ansari2023learning,
  title={Learning Peptide Properties with Positive Examples Only},
  author={Ansari, Mehrad and White, Andrew D},
  journal={bioRxiv},
  pages={2023--06},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
