# DFANet: A Difference Fusion Attention-based method for Semantic Change Detection

[Omkar Oak](https://github.com/omkarsoak)$^{1^*}$, [Rukmini Nazre](https://github.com/rukmini-17)$^1$, [Rujuta Budke](https://github.com/rujuta13)$^1$, Suraj Sawant $^1$, [Gaurav Bhole](https://github.com/Gaurav2543)$^2$ <br>
$^1$ Department of Computer Science and Engineering, COEP Technological University, Pune, India. <br>
$^2$ International Institute of Information Technology, Hyderabad, India. <br>

## Abstract

Semantic Change Detection (SCD) in remote sensing imagery has emerged as a critical technique for monitoring detailed landscape transitions across diverse environments. Despite advances in deep learning approaches including CNNs and Transformers, existing SCD methods struggle with inefficient feature integration, limited receptive fields, sub-optimal fusion of temporal information, and high computational complexity for high-resolution imagery. This paper proposes a Difference Fusion Attention Network (DFANet), a novel multi-task learning framework for SCD in remote sensing imagery. DFANet employs a dual encoder decoder architecture with dedicated paths for Land Cover Mapping (LCM) and Change Detection (CD). The Difference Fusion module leverages channel attention mechanisms to highlight meaningful changes while suppressing noise in feature differences, enabling more effective detection of land cover transitions between temporal images. We incorporate attention mechanisms to focus on informative regions and dense feature connections for improved information flow. We integrate LCM with CD through a refinement module that leverages complementary information between tasks. To address class imbalance challenges, we employ a specialized loss function combining cross-entropy and Tversky loss with class-weighted balancing. Our approach achieves mIoU scores of 73.30% and 74.12%, on TERRA-CD and SECOND datasets respectively, outperforming existing methods. DFANet establishes a new benchmark for SCD in remote sensing applications. Code will be made available.

## Citation

If you use this code or the findings in your research, please cite our paper:

```
Coming Soon!
```

## Contact

For any questions or inquiries about this research, please open an issue on this repository or contact the corresponding author.
