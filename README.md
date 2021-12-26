# Image Clustering with Variational Auto Encoder

Processing Stages:
 - Raw images were received in MRC format. Each MRC file have stack of images belonging to a thread of protein partial. Number of images in each MRC file could vary based on thread length but image dimension would be fixed.
 - As a first step MRC files are converted to numpy array and stored in h5 files, they have similar structure as that of MRC files. In this stage images were scaled to 0 to 1 range from 0 to 255. **ChepVAE001.ipynb** does this job.
