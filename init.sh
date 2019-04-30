#!/bin/bash
/usr/bin/anaconda/envs/py35/bin/python3 -m nltk.downloader -d /home/nltk_data popular
# echo 'PYSPARK_PYTHON=/usr/bin/anaconda/envs/py35/bin/python3' >> /etc/environment
# echo 'PYSPARK_DRIVER_PYTHON=/usr/bin/anaconda/envs/py35/bin/python3' >> /etc/environment