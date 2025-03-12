# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    This script demonstrates a pipeline for cardiac MR image analysis.
    """
import os
import urllib.request
import shutil

if __name__ == '__main__':
    # The GPU device id
    CUDA_VISIBLE_DEVICES = 0
    
    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'
    
    # Download demo images
    print('Downloading demo images ...')
    for i in [1, 2]:
        if not os.path.exists('demo_image/{0}'.format(i)):
            os.makedirs('demo_image/{0}'.format(i))
        for seq_name in ['sa']:
            f = 'demo_image/{0}/{1}.nii.gz'.format(i, 'sa')
            urllib.request.urlretrieve(URL + f, f)

    # Download information spreadsheet
    print('Downloading information spreadsheet ...')
    if not os.path.exists('demo_csv'):
        os.makedirs('demo_csv')
    for f in ['demo_csv/blood_pressure_info.csv']:
        urllib.request.urlretrieve(URL + f, f)
    
    # Download trained models
    print('Downloading trained models ...')
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    for model_name in ['FCN_sa']:
        for f in ['trained_model/{0}.meta'.format(model_name),
                  'trained_model/{0}.index'.format(model_name),
                  'trained_model/{0}.data-00000-of-00001'.format(model_name)]:
            urllib.request.urlretrieve(URL + f, f)
    
    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image segmentation')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('python common/deploy_network.py --seq_name sa --data_dir demo_image --model_path trained_model/FCN_sa')
    
    print('Done.')
