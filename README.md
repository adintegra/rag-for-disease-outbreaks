## MAS Master Thesis

**How can multi-modal data fusion inform and enhance the prediction and reporting of malaria outbreaks?**

- [MAS Master Thesis](#mas-master-thesis)
- [Introduction](#introduction)
- [Planning](#planning)
- [Code Environment](#code-environment)
- [Data Acquisition](#data-acquisition)
  - [Approach 1: RAG](#approach-1-rag)
- [References](#references)


## Introduction

/Users/mark/Documents/ownCloud/documents/education/MAS Data Science


## Planning

TODO: embed a project plan


## Code Environment

On MacOS you must install a native ARM build if you are running on Apple Silicon (Mn processors). Otherwise, Python will default to x86 builds which will run on Rosetta and ML will not run at all. See also [here](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple).


```
CONDA_SUBDIR=osx-arm64 conda create --name pg-vector-rag python=3.12 -c conda-forge

conda env remove --name pg-vector-rag

pip install -r requirements.txt
```

## Data Acquisition

Data will be found in the [./data](./data/) folder.

### Approach 1: RAG




## References

- [Reliable, fully local RAG agents with LLaMA3.2-3b](https://www.youtube.com/watch?v=bq1Plo2RhYI)
-