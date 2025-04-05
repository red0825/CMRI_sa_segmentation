from biobank_utils import *

if __name__ == '__main__':
    eid_list = [...]  # subject EIDs
    util_dir = '/path/to/util'
    ukbkey_path = '/path/to/ukb.key'
    data_root = '/path/to/raw_data'
    output_root = '/path/to/output'
    annotation_dir = '/path/to/annotation'
    annotation_cache_dir = '/path/to/annotation_cache'
    log_dir = '/path/to/logs'
    field_list = [20208,20209]
    process_num = 2

    # Step 1: Download
    failed = parallel_download(eid_list, util_dir, ukbkey_path, data_root, log_dir, field_list, process_num)

    # Step 2: Retry if needed
    if failed:
        retry_failed_eids(log_dir, retry_times=2, util_dir=util_dir, ukbkey_path=ukbkey_path,
                          data_root=data_root, field_list=field_list, process_num=process_num)

    # Step 3: Process all zip
    parallel_process_zip(data_root, annotation_dir, annotation_cache_dir, output_root, process_num)