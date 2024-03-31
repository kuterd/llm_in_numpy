# Implementation of Microsoft's Phi-2 model in Numpy

This is a simple and well documented implementation of Microsoft's Phi-2 model in Numpy, Phi-2 architecture is not hugely different from Meta's Llama models but phi-2 was chosen because it's only 2 billion parameters and will comfortable fit into RAM in many systems. You will need around 5 GBs of CPU RAM to run this code. 

## Instal dependencies

```bash
    pip install -r requirements.txt
```

## Download the Phi-2 weights from Hugging Face

```bash
git clone https://huggingface.co/microsoft/phi-2 
git lfs fetch --all
```

## Run
   `python3 phi_numpy.py --progress` 
    NOTE: This implementation is written for educational purposes and is not optimized for performance, it may take more than 20 seconds per token.

