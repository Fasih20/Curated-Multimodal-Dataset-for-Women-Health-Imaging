"""
Source: original Colab notebook, cell index [80]
Auto-extracted -- review before treating as final.
"""

import shutil
from google.colab import files

# The name of your folder
# folder_to_download = "pipeline_data"
folder_to_download = "pipeline_data"

# 1. Zip the folder
# This creates 'comparison_report.zip' in your current Colab directory
shutil.make_archive(folder_to_download, 'zip', folder_to_download)

# 2. Trigger the download to your local machine
files.download(f"{folder_to_download}.zip")