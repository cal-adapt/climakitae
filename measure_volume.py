#!/usr/bin/env python3
"""
Parallel AWS S3 bucket evaluation script.

This script uses concurrent.futures to parallelize AWS S3 ls operations
and provides summarized, human-readable output similar to:
aws s3 ls s3://bucket-name --summarize --human-readable --recursive
"""

import concurrent.futures
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

timeout = 30000  # Default timeout in seconds


def run_s3_ls_command(s3_path: str) -> Tuple[str, Dict[str, any], str]:
    """Run AWS S3 ls command for a specific S3 path.

    Parameters
    ----------
    s3_path : str
        Full S3 path to list (e.g., s3://bucket/folder)

    Returns
    -------
    Tuple[str, Dict[str, any], str]
        (s3_path, summary_dict, error_message) from the command
    """
    cmd = [
        "aws",
        "s3",
        "ls",
        s3_path,
        "--summarize",
        "--human-readable",
        "--recursive",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.stderr:
            return s3_path, {}, result.stderr
        else:
            # Parse and return only summary data
            summary = parse_s3_output(result.stdout)
            return s3_path, summary, ""
    except subprocess.TimeoutExpired:
        return s3_path, {}, f"Timeout after {timeout} seconds"
    except Exception as e:
        return s3_path, {}, f"Error: {str(e)}"


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """Parse S3 URL to extract bucket and prefix.

    Parameters
    ----------
    s3_url : str
        S3 URL in format s3://bucket-name or s3://bucket-name/prefix

    Returns
    -------
    Tuple[str, str]
        (bucket_name, prefix) where prefix may be empty string
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("S3 URL must start with 's3://'")

    # Remove s3:// prefix
    path = s3_url[5:]

    # Split bucket and prefix
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket, prefix


def get_s3_folders(s3_url: str, max_depth: int = 2) -> List[str]:
    """Get list of S3 folder paths to process in parallel.

    Parameters
    ----------
    s3_url : str
        Base S3 URL (e.g., s3://bucket-name)
    max_depth : int
        Maximum directory depth to parallelize

    Returns
    -------
    List[str]
        List of full S3 paths to process
    """
    bucket, base_prefix = parse_s3_url(s3_url)

    def _get_folders_recursive(current_prefix: str, current_depth: int) -> List[str]:
        """Recursively discover folders up to max_depth.

        Parameters
        ----------
        current_prefix : str
            Current prefix being explored
        current_depth : int
            Current depth level (0 = root)

        Returns
        -------
        List[str]
            List of folder paths at this level and below
        """
        if current_depth >= max_depth:
            # At max depth, return current path as a processing unit
            if current_prefix:
                return [f"s3://{bucket}/{current_prefix}"]
            else:
                return [f"s3://{bucket}"]

        # Build the ls command for current level
        if current_prefix:
            cmd = ["aws", "s3", "ls", f"s3://{bucket}/{current_prefix}/"]
        else:
            cmd = ["aws", "s3", "ls", f"s3://{bucket}/"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                print(
                    f"Error getting folders at depth {current_depth}: {result.stderr}"
                )
                # Return current level as fallback
                if current_prefix:
                    return [f"s3://{bucket}/{current_prefix}"]
                else:
                    return [f"s3://{bucket}"]

            folder_paths = []
            found_folders = False

            for line in result.stdout.strip().split("\n"):
                if line.strip() and "PRE" in line:
                    found_folders = True
                    # Extract directory name
                    folder_name = line.split()[-1].rstrip("/")

                    # Build new prefix
                    if current_prefix:
                        new_prefix = f"{current_prefix}/{folder_name}"
                    else:
                        new_prefix = folder_name

                    # Recursively get subfolders
                    subfolder_paths = _get_folders_recursive(
                        new_prefix, current_depth + 1
                    )
                    folder_paths.extend(subfolder_paths)

            # If no folders found at this level, return current path
            if not found_folders:
                if current_prefix:
                    return [f"s3://{bucket}/{current_prefix}"]
                else:
                    return [f"s3://{bucket}"]

            return folder_paths

        except Exception as e:
            print(f"Error getting folders at depth {current_depth}: {e}")
            # Return current level as fallback
            if current_prefix:
                return [f"s3://{bucket}/{current_prefix}"]
            else:
                return [f"s3://{bucket}"]

    # Start recursive discovery from base prefix
    return _get_folders_recursive(base_prefix, 0)


def parse_s3_output(output: str) -> Dict[str, any]:
    """Parse S3 ls output to extract summary information.

    Parameters
    ----------
    output : str
        Raw output from aws s3 ls command

    Returns
    -------
    Dict[str, any]
        Dictionary with parsed summary info
    """
    summary = {"total_objects": 0, "total_size_str": "0.0 Bytes"}

    lines = output.strip().split("\n")
    for line in lines:
        if "Total Objects:" in line:
            try:
                summary["total_objects"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Total Size:" in line:
            # Extract size (already human readable from AWS)
            try:
                size_str = line.split(":", 1)[1].strip()
                summary["total_size_str"] = size_str
            except IndexError:
                pass

    return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Parallel AWS S3 bucket evaluation")
    parser.add_argument(
        "s3_url",
        help="S3 URL to evaluate (e.g., s3://bucket-name or s3://bucket-name/prefix)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum directory depth for parallelization (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=timeout,
        help="Timeout per operation in seconds (default: 30000 seconds)",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed results to file (default: only summary)",
    )

    args = parser.parse_args()

    # Parse and validate S3 URL
    try:
        bucket, prefix = parse_s3_url(args.s3_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Starting parallel S3 evaluation of: {args.s3_url}")
    print(f"Bucket: {bucket}, Prefix: {prefix or '(root)'}")
    print(f"Workers: {args.workers}, Max depth: {args.max_depth}")
    print(f"Start time: {datetime.now()}")
    print("-" * 50)

    # Get folders to process
    print("Discovering S3 folders...")
    folder_paths = get_s3_folders(args.s3_url, args.max_depth)
    print(f"Found {len(folder_paths)} folders to process:")
    for path in folder_paths:
        print(f"  - {path}")

    # Process folders in parallel - only keep summary data
    folder_summaries = {}  # path -> summary dict
    total_objects = 0
    total_errors = 0
    error_paths = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(run_s3_ls_command, path): path for path in folder_paths
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]

            try:
                path_result, summary, error = future.result(timeout=args.timeout)

                if error:
                    print(f"ERROR in {path_result}: {error}")
                    total_errors += 1
                    error_paths.append((path_result, error))
                else:
                    total_objects += summary.get("total_objects", 0)
                    folder_summaries[path_result] = summary

                    print(
                        f"✓ {path_result}: {summary.get('total_objects', 0):,} objects"
                    )
                    if "total_size_str" in summary:
                        print(f"  Size: {summary['total_size_str']}")

            except concurrent.futures.TimeoutError:
                print(f"TIMEOUT: {path} exceeded {args.timeout} seconds")
                total_errors += 1
                error_paths.append((path, f"Timeout after {args.timeout} seconds"))
            except Exception as e:
                print(f"EXCEPTION in {path}: {e}")
                total_errors += 1
                error_paths.append((path, str(e)))

    # Final summary
    print("-" * 50)
    print(f"Parallel S3 evaluation completed")
    print(f"End time: {datetime.now()}")
    print(f"Total objects found: {total_objects:,}")
    print(f"Total errors: {total_errors}")
    print(f"Folders processed: {len(folder_paths)}")

    # Write compact summary to file
    safe_bucket_name = bucket.replace("/", "_")
    output_file = Path(
        f"s3_summary_{safe_bucket_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(output_file, "w") as f:
        f.write("AWS S3 Bucket Evaluation Summary\n")
        f.write(f"S3 URL: {args.s3_url}\n")
        f.write(f"Bucket: {bucket}\n")
        f.write(f"Prefix: {prefix or '(root)'}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"Total Objects: {total_objects:,}\n")
        f.write(f"Total Errors: {total_errors}\n")
        f.write("=" * 50 + "\n\n")

        # Write folder summaries
        f.write("FOLDER SUMMARIES:\n")
        for path, summary in folder_summaries.items():
            f.write(
                f"{path}: {summary['total_objects']:,} objects, {summary['total_size_str']}\n"
            )

        # Write errors if any
        if error_paths:
            f.write(f"\nERRORS ({len(error_paths)}):\n")
            for path, error in error_paths:
                f.write(f"{path}: {error}\n")

    print(f"Summary written to: {output_file}")

    # Exit with error code if there were failures
    if total_errors > 0:
        print(f"Warning: {total_errors} folders failed to process")
        sys.exit(1)
    else:
        print("All folders processed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
