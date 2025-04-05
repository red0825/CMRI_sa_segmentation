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

import os
import re
import cv2
import pydicom as dicom
import SimpleITK as sitk
import nibabel as nib
import pickle
import numpy as np
from xml.dom import minidom
import subprocess
import zipfile
from zipfile import ZipFile
import time
import multiprocessing
from tqdm import tqdm
from functools import partial
from datetime import datetime
import shutil
import pandas as pd
import glob

def keepElementNodes(nodes):
    """ Get the element nodes """
    nodes2 = []
    for node in nodes:
        if node.nodeType == node.ELEMENT_NODE:
            nodes2 += [node]
    return nodes2


def parseContours(node):
    """
        Parse a Contours object. Each Contours object may contain several contours.
        We first parse the contour name, then parse the points and pixel size.
        """
    contours = {}
    for child in keepElementNodes(node.childNodes):
        contour_name = child.getAttribute('Hash:key')
        sup = 1
        for child2 in keepElementNodes(child.childNodes):
            if child2.getAttribute('Hash:key') == 'Points':
                points = []
                for child3 in keepElementNodes(child2.childNodes):
                    x = float(child3.getElementsByTagName('Point:x')[0].firstChild.data)
                    y = float(child3.getElementsByTagName('Point:y')[0].firstChild.data)
                    points += [[x, y]]
            if child2.getAttribute('Hash:key') == 'SubpixelResolution':
                sub = int(child2.firstChild.data)
        points = np.array(points)
        points /= sub
        contours[contour_name] = points
    return contours


def traverseNode(node, uid_contours):
    """ Traverse the nodes """
    child = node.firstChild
    while child:
        if child.nodeType == child.ELEMENT_NODE:
            # This is where the information for each dicom file starts
            if child.getAttribute('Hash:key') == 'ImageStates':
                for child2 in keepElementNodes(child.childNodes):
                    # UID for the dicom file
                    uid = child2.getAttribute('Hash:key')
                    for child3 in keepElementNodes(child2.childNodes):
                        if child3.getAttribute('Hash:key') == 'Contours':
                            contours = parseContours(child3)
                            if contours:
                                uid_contours[uid] = contours
        traverseNode(child, uid_contours)
        child = child.nextSibling


def parseFile(xml_name, output_dir):
    """ Parse a cvi42 xml file """
    dom = minidom.parse(xml_name)
    uid_contours = {}
    traverseNode(dom, uid_contours)

    # Save the contours for each dicom file
    for uid, contours in uid_contours.items():
        with open(os.path.join(output_dir, '{0}.pickle'.format(uid)), 'wb') as f:
            pickle.dump(contours, f)

def download_eid(eid, util_dir, ukbkey_path, data_root, log_dir, field_list):
    eid = str(eid)
    data_dir = os.path.join(data_root, eid)
    dicom_dir = os.path.join(data_dir, 'dicom')
    os.makedirs(dicom_dir, exist_ok=True)

    batch_file = os.path.join(data_dir, f"{eid}_batch")
    with open(batch_file, 'w') as f_batch:
        for field_id in field_list:
            f_batch.write(f"{eid} {field_id}\n")

    ukbfetch_exec = os.path.join(util_dir, 'ukbfetch')
    cmd = f"{ukbfetch_exec} -b{batch_file} -a{ukbkey_path}"

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300
        )

        # Check if valid zip exists
        zip_files = [f for f in os.listdir(dicom_dir) if f.endswith('.zip')]
        valid_zips = [f for f in zip_files if zipfile.is_zipfile(os.path.join(dicom_dir, f))]

        elapsed = time.time() - start_time

        if result.returncode != 0 or not valid_zips:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{eid}.log")
            with open(log_path, 'w') as log_file:
                log_file.write(f"[COMMAND] {cmd}\n\n")
                log_file.write(result.stdout.decode(errors='ignore'))
                log_file.write(f"\n[ERROR] Download failed or zip invalid. returncode={result.returncode}, zip count={len(valid_zips)}\n")
            return eid, elapsed, False
        else:
            return eid, elapsed, True

    except Exception as e:
        elapsed = time.time() - start_time
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{eid}.log")
        with open(log_path, 'w') as log_file:
            log_file.write(f"[COMMAND] {cmd}\n\n")
            log_file.write(f"[EXCEPTION] {str(e)}\n")
        return eid, elapsed, False


def parallel_download(eid_list, util_dir, ukbkey_path, data_root, log_dir, field_list, process_num=32):
    print(f"Total subjects to download: {len(eid_list)}")
    print(f"Using {process_num} processes")

    with multiprocessing.Pool(processes=process_num) as pool:
        func = partial(
            download_eid,
            util_dir=util_dir,
            ukbkey_path=ukbkey_path,
            data_root=data_root,
            log_dir=log_dir,
            field_list=field_list
        )
        results = list(tqdm(pool.imap(func, eid_list), total=len(eid_list)))

    # Extract failed list
    failed = [eid for eid, _, success in results if not success]
    total_time = sum([elapsed for _, elapsed, _ in results])
    avg_time = total_time / len(eid_list) if eid_list else 0

    if failed:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        failed_log_file = os.path.join(log_dir, f'failed_eids_{timestamp}.txt')
        with open(failed_log_file, 'w') as f:
            for eid in failed:
                f.write(f"{eid}\n")

    # Summary
    print("\nDownload summary:")
    print(f"Success: {len(eid_list) - len(failed)} / {len(eid_list)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed list saved to: {failed_log_file}")
    print(f"Avg download time: {avg_time:.2f} sec")

def retry_failed_eids(log_dir, retry_times, util_dir, ukbkey_path, data_root, field_list, process_num=32):
    # Find latest failed log file
    failed_logs = sorted([f for f in os.listdir(log_dir) if f.startswith('failed_eids_')], reverse=True)
    if not failed_logs:
        print("No failed log file found.")
        return []

    failed_log_file = os.path.join(log_dir, failed_logs[0])
    with open(failed_log_file, 'r') as f:
        failed_eids = [line.strip() for line in f.readlines() if line.strip()]

    for attempt in range(1, retry_times + 1):
        print(f"\n[Retry Attempt {attempt}] Retrying {len(failed_eids)} failed EIDs...")
        failed_eids = parallel_download(failed_eids, util_dir, ukbkey_path, data_root, log_dir, field_list, process_num)

        if not failed_eids:
            print("All retries succeeded.")
            return []
    print("Retry limit reached.")
    return failed_eids

def process_single_zip(zip_path, output_root, annotation_dir, annotation_cache_dir):
    try:
        eid_field = os.path.basename(zip_path).split('.')[0]
        eid = eid_field.split('_')[0]
        data_dir = os.path.join(output_root, eid)
        dicom_dir = os.path.join(data_dir, f'{eid_field}_dicom')
        os.makedirs(dicom_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Unzip
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_dir)

        # Manifest process
        if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
            shutil.copy(os.path.join(dicom_dir, 'manifest.cvs'),
                        os.path.join(dicom_dir, 'manifest.csv'))

        process_manifest(
            os.path.join(dicom_dir, 'manifest.csv'),
            os.path.join(dicom_dir, 'manifest2.csv')
        )

        df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'))
        for series_name, series_df in df2.groupby('series discription'):
            series_dir = os.path.join(dicom_dir, series_name)
            os.makedirs(series_dir, exist_ok=True)
            for f in series_df['filename']:
                shutil.move(os.path.join(dicom_dir, f), series_dir)

        # Annotation process
        cvi42_contours_dir = os.path.join(data_dir, f'{eid}_contours')
        xml_name = os.path.join(annotation_cache_dir, f'{eid}.cvi42wsx')
        annotation_zip = os.path.join(annotation_dir, f'{eid}.zip')

        if os.path.exists(annotation_zip) and not os.path.exists(xml_name):
            with ZipFile(annotation_zip, 'r') as zip_ref:
                zip_ref.extractall(annotation_cache_dir)

            os.makedirs(cvi42_contours_dir, exist_ok=True)
            parseFile(xml_name, cvi42_contours_dir)

        # Convert to NIfTI
        dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir if os.path.exists(cvi42_contours_dir) else None)
        dset.read_dicom_images()
        dset.convert_dicom_to_nifti(data_dir)

        shutil.rmtree(dicom_dir, ignore_errors=True)
        return eid
    except Exception as e:
        print(f"[ERROR] Failed on {zip_path}: {e}")
        return None

def parallel_process_zip(data_root, annotation_dir, annotation_cache_dir, output_root, process_num=32):
    zip_files = []
    for eid in os.listdir(data_root):
        dicom_dir = os.path.join(data_root, eid, 'dicom')
        if os.path.exists(dicom_dir):
            for f in os.listdir(dicom_dir):
                if f.endswith('.zip'):
                    zip_files.append(os.path.join(dicom_dir, f))

    print(f"[INFO] Found {len(zip_files)} zip files to process.")
    func = partial(
        process_single_zip,
        output_root=output_root,
        annotation_dir=annotation_dir,
        annotation_cache_dir=annotation_cache_dir
    )
    with multiprocessing.Pool(process_num) as pool:
        results = list(tqdm(pool.imap(func, zip_files), total=len(zip_files)))
    

    for eid in set(filter(None, results)):
        xml_path = os.path.join(annotation_cache_dir, f'{eid}.cvi42wsx')
        contour_path = os.path.join(output_root, eid, f'{eid}_contours')
        if os.path.exists(xml_path):
            os.remove(xml_path)
        for path in glob.glob(contour_path):
            shutil.rmtree(path, ignore_errors=True)

def repl(m):
    """ Function for reformatting the date """
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))


def process_manifest(name, name2):
    """
        Read the lines in the manifest.csv file and check whether the date format contains
        a comma, which needs to be removed since it causes problems in parsing the file.
        """
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)


class BaseImage(object):
    """ Representation of an image by an array, an image-to-world affine matrix and a temporal spacing """
    volume = np.array([])
    affine = np.eye(4)
    dt = 1

    def WriteToNifti(self, filename):
        nim = nib.Nifti1Image(self.volume, self.affine)
        nim.header['pixdim'][4] = self.dt
        nim.header['sform_code'] = 1
        nib.save(nim, filename)


class Biobank_Dataset(object):
    """ Class for managing Biobank datasets """
    def __init__(self, input_dir, xml_dir=None):
        """
            Initialise data
            This is important, otherwise the dictionaries will not be cleaned between instances.
            """
        self.subdir = {}
        self.data = {}

        # Find and sort the DICOM sub directories
        subdirs = sorted(os.listdir(input_dir))
        sax_dir = []
        lax_2ch_dir = []
        lax_3ch_dir = []
        lax_4ch_dir = []
        sax_mix_dir = []
        lax_mix_dir = []
        ao_dir = []
        lvot_dir = []
        flow_dir = []
        flow_mag_dir = []
        flow_pha_dir = []
        shmolli_dir = []
        shmolli_fitpar_dir = []
        shmolli_t1map_dir = []
        tag_dir = []
        for s in subdirs:
            m = re.match('CINE_segmented_SAX_b(\d*)$', s)
            if m:
                sax_dir += [(os.path.join(input_dir, s), int(m.group(1)))]
            elif re.match('CINE_segmented_LAX_2Ch$', s):
                lax_2ch_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_LAX_3Ch$', s):
                lax_3ch_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_LAX_4Ch$', s):
                lax_4ch_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_SAX$', s):
                sax_mix_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_LAX$', s):
                lax_mix_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_Ao_dist$', s):
                ao_dir = os.path.join(input_dir, s)
            elif re.match('CINE_segmented_LVOT$', s):
                lvot_dir = os.path.join(input_dir, s)
            elif re.match('flow_250_tp_AoV_bh_ePAT@c$', s):
                flow_dir = os.path.join(input_dir, s)
            elif re.match('flow_250_tp_AoV_bh_ePAT@c_MAG$', s):
                flow_mag_dir = os.path.join(input_dir, s)
            elif re.match('flow_250_tp_AoV_bh_ePAT@c_P$', s):
                flow_pha_dir = os.path.join(input_dir, s)
            elif re.match('ShMOLLI_192i_SAX_b2s$', s):
                shmolli_dir = os.path.join(input_dir, s)
            elif re.match('ShMOLLI_192i_SAX_b2s_SAX_b2s_FITPARAMS$', s):
                shmolli_fitpar_dir = os.path.join(input_dir, s)
            elif re.match('ShMOLLI_192i_SAX_b2s_SAX_b2s_SAX_b2s_T1MAP$', s):
                shmolli_t1map_dir = os.path.join(input_dir, s)
            m = re.match('cine_tagging_3sl_SAX_b(\d*)s$', s)
            if m:
                tag_dir += [(os.path.join(input_dir, s), int(m.group(1)))]

        if not sax_dir:
            print('Warning: SAX subdirectories not found!')
            if sax_mix_dir:
                print('But a mixed SAX directory has been found. '
                      'We will sort it into directories for each slice.')
                list = sorted(os.listdir(sax_mix_dir))
                d = dicom.read_file(os.path.join(sax_mix_dir, list[0]))
                T = d.CardiacNumberOfImages
                Z = int(np.floor(len(list) / float(T)))
                for z in range(Z):
                    s = os.path.join(input_dir, 'CINE_segmented_SAX_b{0}'.format(z))
                    os.makedirs(s)
                    for f in list[z * T:(z + 1) * T]:
                        os.system('mv {0}/{1} {2}'.format(sax_mix_dir, f, s))
                    sax_dir += [(s, z)]

        if not lax_2ch_dir and not lax_3ch_dir and not lax_4ch_dir:
            #print('Warning: LAX subdirectories not found!')
            if lax_mix_dir:
                print('But a mixed LAX directory has been found. '
                      'We will sort it into directories for 2Ch, 3Ch and 4Ch views.')
                list = sorted(os.listdir(lax_mix_dir))
                d = dicom.read_file(os.path.join(lax_mix_dir, list[0]))
                T = d.CardiacNumberOfImages
                if len(list) != 3 * T:
                    print('Error: cannot split files into three partitions!')
                else:
                    lax_3ch_dir = os.path.join(input_dir, 'CINE_segmented_LAX_3Ch')
                    os.makedirs(lax_3ch_dir)
                    for f in list[:T]:
                        os.system('mv {0}/{1} {2}'.format(lax_mix_dir, f, lax_3ch_dir))

                    lax_4ch_dir = os.path.join(input_dir, 'CINE_segmented_LAX_4Ch')
                    os.makedirs(lax_4ch_dir)
                    for f in list[T:2 * T]:
                        os.system('mv {0}/{1} {2}'.format(lax_mix_dir, f, lax_4ch_dir))

                    lax_2ch_dir = os.path.join(input_dir, 'CINE_segmented_LAX_2Ch')
                    os.makedirs(lax_2ch_dir)
                    for f in list[2 * T:3 * T]:
                        os.system('mv {0}/{1} {2}'.format(lax_mix_dir, f, lax_2ch_dir))

        self.subdir = {}
        if sax_dir:
            sax_dir = sorted(sax_dir, key=lambda x:x[1])
            self.subdir['sa'] = [x for x, y in sax_dir]
        if lax_2ch_dir:
            self.subdir['la_2ch'] = [lax_2ch_dir]
        if lax_3ch_dir:
            self.subdir['la_3ch'] = [lax_3ch_dir]
        if lax_4ch_dir:
            self.subdir['la_4ch'] = [lax_4ch_dir]
        if ao_dir:
            self.subdir['ao'] = [ao_dir]
        if lvot_dir:
            self.subdir['lvot'] = [lvot_dir]
        if flow_dir:
            self.subdir['flow'] = [flow_dir]
        if flow_mag_dir:
            self.subdir['flow_mag'] = [flow_mag_dir]
        if flow_pha_dir:
            self.subdir['flow_pha'] = [flow_pha_dir]
        if shmolli_dir:
            self.subdir['shmolli'] = [shmolli_dir]
        if shmolli_fitpar_dir:
            self.subdir['shmolli_fitpar'] = [shmolli_fitpar_dir]
        if shmolli_t1map_dir:
            self.subdir['shmolli_t1map'] = [shmolli_t1map_dir]
        if tag_dir:
            tag_dir = sorted(tag_dir, key=lambda x: x[1])
            for x, y in tag_dir:
                self.subdir['tag_{0}'.format(y)] = [x]

        self.xml_dir = xml_dir

    def find_series(self, dir_name, T):
        """
            In a few cases, there are two or three time sequences or series within each folder.
            We need to find which series to convert.
            """
        files = sorted(os.listdir(dir_name))
        if len(files) > T:
            # Sort the files according to their series UIDs
            series = {}
            for f in files:
                d = dicom.read_file(os.path.join(dir_name, f))
                suid = d.SeriesInstanceUID
                if suid in series:
                    series[suid] += [f]
                else:
                    series[suid] = [f]

            # Find the series which has been annotated, otherwise use the last series.
            if self.xml_dir:
                find_series = False
                for suid, suid_files in series.items():
                    for f in suid_files:
                        contour_pickle = os.path.join(self.xml_dir, os.path.splitext(f)[0] + '.pickle')
                        if os.path.exists(contour_pickle):
                            find_series = True
                            choose_suid = suid
                            break
                if not find_series:
                    choose_suid = sorted(series.keys())[-1]
            else:
                choose_suid = sorted(series.keys())[-1]
            print('There are multiple series. Use series {0}.'.format(choose_suid))
            files = sorted(series[choose_suid])

        if len(files) < T:
            print('Warning: {0}: Number of files < CardiacNumberOfImages! '
                  'We will fill the missing files using duplicate slices.'.format(dir_name))
        return(files)

    def read_dicom_images(self):
        """ Read dicom images and store them in a 3D-t volume. """
        for name, dir in sorted(self.subdir.items()):
            # Read the image volume
            # Number of slices
            Z = len(dir)

            # Read a dicom file at the first slice to get the temporal information
            # We need the number of images in a sequence to check whether multiple sequences are recorded
            d = dicom.read_file(os.path.join(dir[0], sorted(os.listdir(dir[0]))[0]))
            T = d.CardiacNumberOfImages

            # Read a dicom file from the correct series when there are multiple time sequences
            d = dicom.read_file(os.path.join(dir[0], self.find_series(dir[0], T)[0]))
            X = d.Columns
            Y = d.Rows
            T = d.CardiacNumberOfImages
            dx = float(d.PixelSpacing[1])
            dy = float(d.PixelSpacing[0])
            print("dx: {0}, dy: {0}".format(dx, dy))
            
            # DICOM coordinate (LPS)
            #  x: left
            #  y: posterior
            #  z: superior
            # Nifti coordinate (RAS)
            #  x: right
            #  y: anterior
            #  z: superior
            # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
            # Refer to
            # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
            # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

            # The coordinate of the upper-left voxel of the first and second slices
            pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
            pos_ul[:2] = -pos_ul[:2]

            # Image orientation
            axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
            axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
            axis_x[:2] = -axis_x[:2]
            axis_y[:2] = -axis_y[:2]

            if Z >= 2:
                # Read a dicom file at the second slice
                d2 = dicom.read_file(os.path.join(dir[1], sorted(os.listdir(dir[1]))[0]))
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
            else:
                axis_z = np.cross(axis_x, axis_y)

            # Determine the z spacing
            if hasattr(d, 'SpacingBetweenSlices'):
                dz = float(d.SpacingBetweenSlices)
            elif Z >= 2:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                      'Calculate from two successive slices.')
                dz = float(np.linalg.norm(pos_ul2 - pos_ul))
            else:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                      'Use attribute SliceThickness instead.')
                dz = float(d.SliceThickness)

            # Affine matrix which converts the voxel coordinate to world coordinate
            affine = np.eye(4)
            affine[:3, 0] = axis_x * dx
            affine[:3, 1] = axis_y * dy
            affine[:3, 2] = axis_z * dz
            affine[:3, 3] = pos_ul

            # The 4D volume
            volume = np.zeros((X, Y, Z, T), dtype='float32')
            if self.xml_dir:
                # Save both label map in original resolution and upsampled label map.
                # The image annotation by defaults upsamples the image using xml_dir and then
                # annotate on the upsampled image.
                up = 4
                label = np.zeros((X, Y, Z, T), dtype='int16')
                label_up = np.zeros((X * up, Y * up, Z, T), dtype='int16')

            # Go through each slice
            for z in range(0, Z):
                # In a few cases, there are two or three time sequences or series within each folder.
                # We need to find which seires to convert.
                files = self.find_series(dir[z], T)

                # Now for this series, sort the files according to the trigger time.
                files_time = []
                for f in files:
                    d = dicom.read_file(os.path.join(dir[z], f))
                    t = d.TriggerTime
                    files_time += [[f, t]]
                files_time = sorted(files_time, key=lambda x: x[1])

                # Read the images
                for t in range(0, T):
                    try:
                        f = files_time[t][0]
                        d = dicom.read_file(os.path.join(dir[z], f))
                        volume[:, :, z, t] = d.pixel_array.transpose()
                    except IndexError:
                        print('Warning: dicom file missing for {0}: time point {1}. '
                              'Image will be copied from the previous time point.'.format(dir[z], t))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                    except (ValueError, TypeError):
                        print('Warning: failed to read pixel_array from file {0}. '
                              'Image will be copied from the previous time point.'.format(os.path.join(dir[z], f)))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                    except NotImplementedError:
                        print('Warning: failed to read pixel_array from file {0}. '
                              'pydicom cannot handle compressed dicom files. '
                              'Switch to SimpleITK instead.'.format(os.path.join(dir[z], f)))
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(os.path.join(dir[z], f))
                        img = sitk.GetArrayFromImage(reader.Execute())
                        volume[:, :, z, t] = np.transpose(img[0], (1, 0))

                    if self.xml_dir:
                        # Check whether there is a corresponding xml_dir contour file for this dicom
                        contour_pickle = os.path.join(self.xml_dir, os.path.splitext(f)[0] + '.pickle')
                        if os.path.exists(contour_pickle):
                            with open(contour_pickle, 'rb') as f:
                                contours = pickle.load(f)

                                # Labels
                                lv_endo = 1
                                lv_epi = 2
                                rv_endo = 3
                                la_endo = 1
                                ra_endo = 2

                                ordered_contours = []
                                if 'sarvendocardialContour' in contours:
                                    ordered_contours += [(contours['sarvendocardialContour'], rv_endo)]

                                if 'saepicardialContour' in contours:
                                    ordered_contours += [(contours['saepicardialContour'], lv_epi)]
                                if 'saepicardialOpenContour' in contours:
                                    ordered_contours += [(contours['saepicardialOpenContour'], lv_epi)]

                                if 'saendocardialContour' in contours:
                                    ordered_contours += [(contours['saendocardialContour'], lv_endo)]
                                if 'saendocardialOpenContour' in contours:
                                    ordered_contours += [(contours['saendocardialOpenContour'], lv_endo)]

                                if 'laraContour' in contours:
                                    ordered_contours += [(contours['laraContour'], ra_endo)]

                                if 'lalaContour' in contours:
                                    ordered_contours += [(contours['lalaContour'], la_endo)]

                                lab_up = np.zeros((Y * up, X * up))
                                for c, l in ordered_contours:
                                    coord = np.round(c * up).astype(np.int32)
                                    cv2.fillPoly(lab_up, [coord], l)

                                label_up[:, :, z, t] = lab_up.transpose()
                                label[:, :, z, t] = lab_up[::up, ::up].transpose()

            # Temporal spacing
            dt = (files_time[1][1] - files_time[0][1]) * 1e-3
            print("times :", files_time[1][1], files_time[0][1])
            print("dt :", dt)

            # Store the image
            self.data[name] = BaseImage()
            self.data[name].volume = volume
            self.data[name].affine = affine
            self.data[name].dt = dt

            if self.xml_dir:
                # Only save the label map if it is non-zero
                if np.any(label):
                    self.data['label_' + name] = BaseImage()
                    self.data['label_' + name].volume = label
                    self.data['label_' + name].affine = affine
                    self.data['label_' + name].dt = dt

                if np.any(label_up):
                    self.data['label_up_' + name] = BaseImage()
                    self.data['label_up_' + name].volume = label_up
                    up_matrix = np.array([[1.0/up, 0, 0, 0], [0, 1.0/up, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    self.data['label_up_' + name].affine = np.dot(affine, up_matrix)
                    self.data['label_up_' + name].dt = dt

    def convert_dicom_to_nifti(self, output_dir):
        """ Save the image in nifti format. """
        for name, image in self.data.items():
            image.WriteToNifti(os.path.join(output_dir, '{0}.nii.gz'.format(name)))
