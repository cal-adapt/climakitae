import subprocess
import time
import json
from typing import List, Dict, Any


#!/usr/bin/env python3
"""
Speed test script for different S3 bucket size checking methods.

This script compares the performance of:
1. AWS CLI ls with recursive listing
2. AWS CLI API list-objects-v2 with size calculation
3. s4cmd du command

Usage:
    python bucket_speed_test.py
"""


def run_command(command: List[str]) -> Dict[str, Any]:
    """
    Run a shell command and measure execution time.

    Parameters
    ----------
    command : List[str]
        Command to execute as list of strings.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing execution time, return code, stdout, and stderr.
    """
    start_time = time.time()

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "execution_time": 300.0,
            "return_code": -1,
            "stdout": "",
            "stderr": "Command timed out after 300 seconds",
            "success": False,
        }
    except Exception as e:
        return {
            "execution_time": 0.0,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def test_aws_s3_ls(bucket_name: str) -> Dict[str, Any]:
    """
    Test AWS S3 ls command with recursive listing.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to test.

    Returns
    -------
    Dict[str, Any]
        Test results including timing and output.
    """
    command = [
        "aws",
        "s3",
        "ls",
        f"s3://{bucket_name}",
        "--recursive",
        "--human-readable",
        "--summarize",
    ]

    print(
        f"Testing: aws s3 ls s3://{bucket_name} --recursive --human-readable --summarize"
    )
    return run_command(command)


def test_aws_s3api_list_objects(bucket_name: str) -> Dict[str, Any]:
    """
    Test AWS S3 API list-objects-v2 with size calculation.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to test.

    Returns
    -------
    Dict[str, Any]
        Test results including timing and output.
    """
    command = [
        "aws",
        "s3api",
        "list-objects-v2",
        "--bucket",
        f"s3://{bucket_name}",
        "--query",
        "sum(Contents[].Size)",
    ]

    print(
        f"Testing: aws s3api list-objects-v2 --bucket {bucket_name} --query 'sum(Contents[].Size)'"
    )
    return run_command(command)


def test_s4cmd_du(bucket_name: str) -> Dict[str, Any]:
    """
    Test s4cmd du command.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to test.

    Returns
    -------
    Dict[str, Any]
        Test results including timing and output.
    """
    command = ["s4cmd", "du", f"s3://{bucket_name}"]

    print(f"Testing: s4cmd du s3://{bucket_name}")
    return run_command(command)


def format_time(seconds: float) -> str:
    """
    Format execution time in a human-readable format.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"


def print_results(bucket_name: str, results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print formatted test results.

    Parameters
    ----------
    bucket_name : str
        Name of the bucket tested.
    results : Dict[str, Dict[str, Any]]
        Test results for each method.
    """
    print(f"\n{'='*60}")
    print(f"RESULTS FOR BUCKET: {bucket_name}")
    print(f"{'='*60}")

    # Sort results by execution time (successful tests first)
    successful_results = [(k, v) for k, v in results.items() if v["success"]]
    failed_results = [(k, v) for k, v in results.items() if not v["success"]]

    successful_results.sort(key=lambda x: x[1]["execution_time"])

    print("\nSUCCESSFUL TESTS (sorted by speed):")
    print("-" * 40)

    for i, (method, result) in enumerate(successful_results, 1):
        time_str = format_time(result["execution_time"])
        print(f"{i}. {method:<20} | {time_str:>8}")

        # Show sample output (first few lines)
        if result["stdout"].strip():
            lines = result["stdout"].strip().split("\n")
            sample_lines = lines[:2] if len(lines) > 2 else lines
            for line in sample_lines:
                if line.strip():
                    print(f"   Output: {line.strip()}")

    if failed_results:
        print(f"\nFAILED TESTS:")
        print("-" * 40)
        for method, result in failed_results:
            print(f"❌ {method}")
            if result["stderr"]:
                print(f"   Error: {result['stderr'][:100]}...")


def run_speed_test(bucket_names: List[str]) -> None:
    """
    Run speed tests on all specified buckets.

    Parameters
    ----------
    bucket_names : List[str]
        List of bucket names to test.
    """
    test_methods = {
        "aws_s3_ls": test_aws_s3_ls,
        "s4cmd_du": test_s4cmd_du,
    }

    all_results = {}

    for bucket_name in bucket_names:
        print(f"\n{'#'*60}")
        print(f"TESTING BUCKET: {bucket_name}")
        print(f"{'#'*60}")

        bucket_results = {}

        for method_name, test_function in test_methods.items():
            print(f"\n--- {method_name.upper()} ---")
            result = test_function(bucket_name)
            bucket_results[method_name] = result

            if result["success"]:
                print(f"✅ Completed in {format_time(result['execution_time'])}")
            else:
                print(f"❌ Failed: {result['stderr']}")

        all_results[bucket_name] = bucket_results
        print_results(bucket_name, bucket_results)

    # Print summary across all buckets
    if len(bucket_names) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL BUCKETS")
        print(f"{'='*60}")

        for method in test_methods.keys():
            successful_times = []
            for bucket_name in bucket_names:
                result = all_results[bucket_name][method]
                if result["success"]:
                    successful_times.append(result["execution_time"])

            if successful_times:
                avg_time = sum(successful_times) / len(successful_times)
                min_time = min(successful_times)
                max_time = max(successful_times)
                print(
                    f"{method:<25} | Avg: {format_time(avg_time):>8} | "
                    f"Min: {format_time(min_time):>8} | Max: {format_time(max_time):>8}"
                )


def main():
    """Main function to run the speed tests."""
    # Configure your bucket names here
    bucket_names = [
        "wfclimres/wrf_jsons",  # Small bucket (< 1GB)
        "cadcat/tmp/kc2",  # Medium bucket (1-10GB)
        "cadcat/hmet",  # Large bucket (> 10GB)
    ]

    print("S3 Bucket Size Check Speed Test")
    print("=" * 40)
    print("This script will test the performance of different methods")
    print("for checking S3 bucket sizes.\n")

    print("Buckets to test:")
    for i, bucket in enumerate(bucket_names, 1):
        print(f"  {i}. {bucket}")

    # Run the tests
    run_speed_test(bucket_names)


if __name__ == "__main__":
    main()
