# GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning
Rui Tang1*, Guankun Wang1*, Long Bai1, Huxin Gao1, Jiewen Lai1, Chi Kit Ng1, Jiazheng Wang2, Fan Zhang2, Hongliang Ren†1 <br/>

## Overview
Language-guided grasping has emerged as a promising paradigm for enabling robots to identify and manipulate target objects through natural language instructions, yet it remains highly challenging in cluttered or occluded scenes. Existing methods often rely on multi-stage pipelines that separate object perception and grasping, which leads to limited cross-modal fusion, redundant computation, and poor generalization in cluttered, occluded, or low-texture scenes. To address these limitations, we propose GeoLanG, an end-to-end multi-task framework built upon the CLIP architecture that unifies visual and linguistic inputs into a shared representation space for robust semantic alignment and improved generalization. To enhance target discrimination under occlusion and low-texture conditions, we explore a more effective use of depth information through the Depth-guided Geometric Module (DGGM), which converts depth into explicit geometric priors and injects them into the attention mechanism without additional computational overhead.
In addition, we propose Adaptive Dense Channel Integration, which adaptively balances the contributions of multi-layer features to produce more discriminative and generalizable visual representations.
Extensive experiments on the OCID-VLG dataset, as well as in both simulation and real-world hardware, demonstrate that GeoLanG enables precise and robust language-guided grasping in complex, cluttered environments, paving the way toward more reliable multimodal robotic manipulation in real-world human-centric settings.
<p align="center">
  <img
    width="1000"
    src=".method.png"
  >
</p>


## Citation

If you find  [**GeoLanG**](https://arxiv.org/abs/2602.04231) useful for your research or development, please cite the following:



```latex
@misc{tang2026geolanggeometryawarelanguageguidedgrasping,
      title={GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning}, 
      author={Rui Tang and Guankun Wang and Long Bai and Huxin Gao and Jiewen Lai and Chi Kit Ng and Jiazheng Wang and Fan Zhang and Hongliang Ren},
      year={2026},
      eprint={2602.04231},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.04231}, 
}
