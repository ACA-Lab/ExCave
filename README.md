
# ExCave: Excavating Consistency Across Editing Steps for Effective Multi-Step Image Editing

## 🌟 Overview

**ExCave** is a training-free and hardware-efficient framework designed for multi-step image editing with diffusion models. 

Existing "inversion-denoising" paradigms suffer from image quality degradation due to error accumulation across multiple steps and significant computational redundancy in unchanged background regions. ExCave addresses these challenges by excavating and leveraging the consistency inherent in the editing process.

### Key Contributions:
* **Inversion Sharing**: Reuses consistent features from a single initial inversion across subsequent edits, effectively eliminating error accumulation and preserving image quality.
* **CacheDiff**: A feature caching mechanism that identifies edited regions and reuses features for unchanged backgrounds, removing redundant computations.
* **GPU-Oriented Optimization**: Custom GPU optimizations that translate theoretical computational savings into a  significant reduction in end-to-end latency.

---

## 📊 Performance

ExCave achieves superior background preservation and inference efficiency in multi-turn editing scenarios:

| Method               | Latency (s) ↓ | PSNR (dB) ↑ | Note                           |
| :------------------- | :-----------: | :---------: | :----------------------------- |
| RF-Solver (Baseline) |    108.58     |    8.22     | Quality degrades over steps    |
| **ExCave (Ours)**    |   **26.62**   |  **8.39**   | **4x Faster & Stable Quality** |

---

## 🛠️ Installation

We recommend using Conda to manage your environment:

```bash
# Clone the repository
git clone https://github.com/ACA-Lab/ExCave.git
cd ExCave

# Create and activate the environment
conda env create -f excave.yaml
conda activate excave
```

## 🚀 **Edit Your Own Image**

### **Command Line**
You can run the following scripts to edit your own image:

```bash
cd src
# This script automatically performs edits based on predefined prompts
bash ./run.sh
```

The ```--inject``` refers to the steps of feature sharing in diffusion model, which is highly related to the performance of editing. 

---

## 🙏 Acknowledgements

We would like to express our sincere gratitude to the following open-source projects and organizations for their invaluable contributions and support to this research:

* **[FLUX](https://github.com/black-forest-labs/flux)**: For providing the powerful foundational generative model that serves as the base of our framework.
* **[RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit)**: For their clean and efficient implementation of the Rectified Flow solver, which significantly inspired our inference pipeline.
* **Institutional Support**: This research was conducted through a close collaboration between the **School of Computer Science at Shanghai Jiao Tong University (SJTU)** and the **OPPO Research Institute**. We thank our colleagues and collaborators for their insightful discussions and technical support.

---

<div align="center">
  For any questions regarding the code or the paper, please open an issue or contact the maintainers at <b>qichunyu@sjtu.edu.cn</b>.
</div>
