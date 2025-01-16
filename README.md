

# Lightning Attention 2

This is an implementation of Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models 2401.04658 using PyTorch.


## Installation

I've given instructions for either uv or pip. You can use either of them.



### uv Installation

Checkout the code and run from source: 

```bash
git clone https://github.com/evelynmitchell/lightning-attention.git
cd lightning-attention
uv run --python 3.12 --with torch --with loguru lighting-attention/src/lightning-attention2/main.py --system
```

### pip Installation

Create a virtual environment for python:

```bash
python -m venv .venv
source .venv/bin/activate
```

Checkout the code and install from source: 

```bash
git clone https://github.com/evelynmitchell/lightning-attention.git
cd lightning-attention
pip install . --system
```

# Usage
```bash
git clone https://github.com/evelynmitchell/lightning-attention.git
cd lightning-attention
uv run --python 3.12  lightning-attention/src/lightning-attention2/main.py --system
```

