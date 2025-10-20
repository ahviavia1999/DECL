### A dual-encoder contrastive learning framework for robust and interpretable molecular property prediction

目录介绍：

- gt_pyg.py是Graph Transformer的代码。

- data_tool.py是与分子操作相关的代码，例如划分子结构、构建分子图等。

- dataset.py是训练和微调过程需要用到的Dataset。

- finetune_class.py是分子性质预测下游分类任务的训练代码。

- finetune_reg.py是分子性质预测下游回归任务的训练代码。

- loss.py是多任务损失代码。

- model_new.py是模糊图编码器和多尺度子结构编码器代码。

- pretrain_singleGPU.py是在单GPU上的模型预训练代码。

- pubchemfp.py是用于获取分子的pubchem指纹。

运行环境如下:

```
absl-py @ file:///home/conda/feedstock_root/build_artifacts/absl-py_1751547525079/work
accelerate==0.33.0
aiobotocore==2.21.1
aiohappyeyeballs==2.6.1
aiohttp==3.11.14
aioitertools==0.12.0
aiosignal==1.3.2
anndata==0.11.4
annotated-types==0.7.0
array_api_compat==1.11.2
asttokens==3.0.0
async-timeout==5.0.1
attrs==25.3.0
bayesian-optimization==2.0.3
beautifulsoup4==4.13.4
botocore==1.37.1
Brotli @ file:///croot/brotli-split_1736182456865/work
cellxgene-census==1.15.0
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1759648874697/work/certifi
charset-normalizer @ file:///croot/charset-normalizer_1721748349566/work
cloudpickle==3.1.1
colorama==0.4.6
comm==0.2.2
contourpy==1.3.1
cycler==0.12.1
Cython==3.1.2
dataclasses==0.6
datasets==2.19.2
decorator==5.2.1
deepchem==2.8.0
dill==0.3.8
et_xmlfile==2.0.0
evaluate==0.4.2
exceptiongroup==1.2.2
executing==2.2.0
filelock @ file:///croot/filelock_1700591183607/work
fonttools==4.56.0
frozenlist==1.5.0
fsspec==2024.3.1
fuzzywuzzy==0.18.0
gget==0.29.1
gmpy2 @ file:///croot/gmpy2_1738085463648/work
grpcio @ file:///croot/grpc-split_1742490263735/work
h5py==3.13.0
huggingface-hub==0.30.2
idna==3.10
importlib_metadata @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_importlib-metadata_1747934053/work
ipython==8.36.0
ipywidgets==8.1.6
jedi==0.19.2
Jinja2==3.1.4
jmespath==1.0.1
joblib==1.4.2
jupyterlab_widgets==3.0.14
kiwisolver==1.4.8
legacy-api-wrap==1.4.1
llvmlite==0.44.0
lxml==5.4.0
Markdown @ file:///home/conda/feedstock_root/build_artifacts/markdown_1757093412127/work
MarkupSafe==2.1.5
matplotlib==3.10.1
matplotlib-inline==0.1.7
mkl-service==2.4.0
mkl_fft @ file:///io/mkl313/mkl_fft_1730824109137/work
mkl_random @ file:///io/mkl313/mkl_random_1730823916628/work
mpmath @ file:///croot/mpmath_1690848262763/work
multidict==6.2.0
multiprocess==0.70.16
mysql-connector-python==9.3.0
natsort==8.4.0
networkx==3.3
numba==0.61.2
numpy==1.26.4
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==9.1.0.70
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.21.5
nvidia-nvtx-cu11==11.8.86
openpyxl==3.1.5
packaging==24.2
pandas==2.2.3
parso==0.8.4
patsy==1.0.1
pexpect==4.9.0
pillow @ file:///croot/pillow_1738010226202/work
powerlaw==1.5
prompt_toolkit==3.0.51
propcache==0.3.1
protobuf @ file:///croot/protobuf_1742419705443/work/bazel-bin/python/dist/protobuf-5.29.3-cp310-abi3-linux_x86_64.whl#sha256=e55acfd476c7a1d8544b98df45c7d8e9c425b9caf0abb1f7e7cfa87c2e017bd1
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pyaml==25.1.0
pyarrow==19.0.1
pyarrow-hotfix==0.7
pydantic==2.11.3
pydantic_core==2.33.1
pyg-lib==0.4.0+pt25cu118
Pygments==2.19.1
pynndescent==0.5.13
pyparsing==3.2.3
PySocks @ file:///home/builder/ci_310/pysocks_1640793678128/work
pytdc==1.1.15
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML @ file:///croot/pyyaml_1728657952215/work
rdkit==2023.9.6
regex==2024.11.6
requests @ file:///croot/requests_1730999120400/work
s3fs==2024.3.1
safetensors==0.5.3
scanpy==1.11.1
scikit-learn==1.5.2
scikit-optimize==0.10.2
scipy==1.15.2
seaborn==0.13.2
session-info2==0.1.2
shap==0.47.2
six==1.17.0
slicer==0.0.8
somacore==1.0.11
soupsieve==2.7
stack-data==0.6.3
statsmodels==0.14.4
sympy==1.13.1
tensorboard @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_tensorboard_1752825441/work/tensorboard-2.20.0-py3-none-any.whl#sha256=9dc9f978cb84c0723acf9a345d96c184f0293d18f166bb8d59ee098e6cfaaba6
tensorboard_data_server @ file:///home/conda/feedstock_root/build_artifacts/tensorboard-data-server_1759413119425/work/tensorboard_data_server-0.7.0-py3-none-manylinux2014_x86_64.whl#sha256=8ec18459e223309f6d0d987e0256a30babd90bf53a779c714797a68b70ed1ad8
threadpoolctl==3.6.0
tiledb==0.29.1
tiledbsoma==1.11.4
tokenizers==0.21.1
torch==2.5.0
torch-geometric==2.6.1
torch_cluster==1.6.3+pt25cu118
torch_scatter @ file:///root/con/torch_scatter-2.1.2%2Bpt25cu118-cp310-cp310-linux_x86_64.whl#sha256=45a153f7863b5cba9515fb92ed987cf148fcf8ffcb53c91858c35aa9c99fc706
torch_sparse==0.6.18+pt25cu118
torch_spline_conv==1.2.2+pt25cu118
torchaudio==2.5.0
torchfuzzy==0.0.2
torchvision==0.20.0
tqdm==4.67.1
traitlets==5.14.3
transformers==4.50.3
triton==3.1.0
typing-inspection==0.4.0
typing_extensions @ file:///croot/typing_extensions_1734714854207/work
tzdata==2025.2
umap-learn==0.5.7
urllib3 @ file:///croot/urllib3_1737133630106/work
wcwidth==0.2.13
weightwatcher==0.7.5.5
Werkzeug @ file:///home/conda/feedstock_root/build_artifacts/werkzeug_1733160440960/work
widgetsnbextension==4.0.14
wrapt==1.17.2
xxhash==3.5.0
yarl==1.18.3
zipp @ file:///home/conda/feedstock_root/build_artifacts/zipp_1749421620841/work
```



使用介绍：

模型预训练通过以下语句执行：

```python
python pretrain_singleGPU.py
```

下游分类任务微调通过以下语句执行：

```python
python finetune_class.py
```

下游回归任务微调通过以下语句执行：

```python
python finetune_reg.py
```


