"""
Source: original Colab notebook, cell index [83]
Auto-extracted -- review before treating as final.
"""

!pip install -U transformers huggingface_hub

# Step 2: Retrieve your secret token securely from Colab
from google.colab import userdata
import os

# This automatically loads your token into the environment variables
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')