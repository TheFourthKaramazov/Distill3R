"""
Data Loading and Processing for Distill3r

Complete data pipeline:
1. Data Preparation: datasets.py + download_data.py create manifest from raw data
2. Teacher Cache Generation: export_fast3r.py runs Fast3R inference and saves .npz caches  
3. Training: teacher_dataset.py loads caches, student learns from them

Memory-efficient samplers provided for RTX 4090 deployment.

For unit testing, import modules directly:
    from distill3r.data.teacher_dataset import TeacherCacheDataset
"""

# Keep this minimal for easier testing and debugging
# Import individual modules as needed for unit tests