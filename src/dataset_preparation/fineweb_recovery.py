#!/usr/bin/env python3
"""Recovery script to download missing FineWeb parquet files directly.

This script downloads only the missing parquet files from the FineWeb dataset,
avoiding the metadata resolution issue that causes the main script to fail.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple
import requests
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not found. Please install it with:")
    print("  pip install huggingface-hub")
    sys.exit(1)


def check_missing_files() -> List[Tuple[str, str]]:
    """Check which parquet files are missing from the cache.
    
    Returns:
        List of tuples (shard_name, filename) for missing files
    """
    cache_dir = Path.home() / '.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb'
    snapshot_dir = cache_dir / 'snapshots/9bb295ddab0e05d785b879661af7260fed5140fc/sample/100BT'
    
    missing_files = []
    
    # Check all expected files (15 shards x 10 files each)
    for shard in range(15):  # 000 to 014
        for file_num in range(10):  # 00000 to 00009
            filename = f'{shard:03d}_{file_num:05d}.parquet'
            file_path = snapshot_dir / filename
            
            if not file_path.exists():
                # The path in the repo is sample/100BT/XXX_XXXXX.parquet
                repo_path = f'sample/100BT/{filename}'
                missing_files.append((repo_path, filename))
    
    return missing_files


def download_missing_files(missing_files: List[Tuple[str, str]]) -> bool:
    """Download missing parquet files directly from HuggingFace.
    
    Args:
        missing_files: List of (repo_path, filename) tuples
        
    Returns:
        True if all downloads successful, False otherwise
    """
    if not missing_files:
        print("✓ No missing files to download!")
        return True
    
    print(f"\nFound {len(missing_files)} missing files:")
    for _, filename in missing_files[:5]:
        print(f"  - {filename}")
    if len(missing_files) > 5:
        print(f"  ... and {len(missing_files) - 5} more")
    
    print(f"\nDownloading missing files...")
    
    success_count = 0
    failed_files = []
    
    for repo_path, filename in tqdm(missing_files, desc="Downloading"):
        try:
            # Download the file using huggingface_hub
            # This will automatically place it in the correct cache location
            local_path = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb",
                filename=repo_path,
                repo_type="dataset",
                revision="9bb295ddab0e05d785b879661af7260fed5140fc"
            )
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Failed to download {filename}: {e}")
            failed_files.append(filename)
            # Continue with other files
    
    print(f"\n✓ Successfully downloaded {success_count}/{len(missing_files)} files")
    
    if failed_files:
        print(f"\n⚠️  Failed files:")
        for f in failed_files:
            print(f"  - {f}")
        return False
    
    return True


def verify_all_files_present() -> bool:
    """Verify that all 150 parquet files are now present.
    
    Returns:
        True if all files present, False otherwise
    """
    cache_dir = Path.home() / '.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb'
    snapshot_dir = cache_dir / 'snapshots/9bb295ddab0e05d785b879661af7260fed5140fc/sample/100BT'
    
    total_expected = 150  # 15 shards x 10 files
    present_count = 0
    
    for shard in range(15):
        for file_num in range(10):
            filename = f'{shard:03d}_{file_num:05d}.parquet'
            file_path = snapshot_dir / filename
            if file_path.exists():
                present_count += 1
    
    print(f"\nVerification: {present_count}/{total_expected} files present")
    
    if present_count == total_expected:
        print("✓ All files successfully downloaded!")
        return True
    else:
        print(f"⚠️  Still missing {total_expected - present_count} files")
        return False


def main():
    """Main recovery process."""
    print("FineWeb Dataset Recovery Tool")
    print("=" * 60)
    print("This tool downloads only the missing parquet files")
    print("preserving your existing 274GB of downloaded data.")
    print("=" * 60)
    
    # Step 1: Check missing files
    print("\nStep 1: Checking for missing files...")
    missing_files = check_missing_files()
    
    if not missing_files:
        print("✓ No missing files found! Dataset is complete.")
        
        # Verify all files are present
        if verify_all_files_present():
            print("\n✅ Dataset ready for processing!")
            print("You can now run:")
            print("  python3 -m src.dataset_preparation.fineweb_download")
            return 0
    
    # Step 2: Download missing files
    print(f"\nStep 2: Downloading {len(missing_files)} missing files...")
    print("This should only take a few minutes...")
    
    success = download_missing_files(missing_files)
    
    # Step 3: Verify completion
    print("\nStep 3: Verifying dataset completeness...")
    if verify_all_files_present():
        print("\n✅ Recovery complete! Dataset ready for processing.")
        print("\nYou can now run:")
        print("  python3 -m src.dataset_preparation.fineweb_download")
        return 0
    else:
        print("\n⚠️  Recovery incomplete. Some files may still be missing.")
        print("You can try running this script again or use the main download script.")
        return 1


if __name__ == "__main__":
    sys.exit(main())