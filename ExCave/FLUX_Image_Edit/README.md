<div align="center">
  
# Excavating Consistency Across Editing Steps for Effective Multi-Step Image Editing

</div>




# Code Setup
Running the following command to construct the environment.
```
conda env create -f excave.yaml
conda activate excave
```


# Edit Your Own Image

## Command Line
You can run the following scripts to edit your own image.: 
```
cd src
bash ./run.sh
```

The ```--inject``` refers to the steps of feature sharing in diffusion model, which is highly related to the performance of editing. 

# Acknowledgements
We thank [FLUX](https://github.com/black-forest-labs/flux/tree/main) and [RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit) for their clean codebase.