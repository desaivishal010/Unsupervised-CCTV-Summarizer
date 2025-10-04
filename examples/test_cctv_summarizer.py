#!/usr/bin/env python3
"""
Test script for CCTV Summarization System
Tests the system on a single video file with comprehensive analysis.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from cctv_summarizer import CCTVAnalyzer


def create_test_video(output_path: str, duration: int = 30) -> str:
    """
    Create a synthetic test video with various motion patterns for testing.
    
    Args:
        output_path: Path where to save the test video
        duration: Duration of test video in seconds
        
    Returns:
        Path to created test video
    """
    print(f"Creating synthetic test video ({duration}s)...")
    
    # Video properties
    fps = 30
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frames with different motion patterns
    for frame_num in range(total_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark background
        
        t = frame_num / fps  # Time in seconds
        
        # Add different motion patterns at different times
        if 5 < t < 10:  # Slow movement
            x = int(100 + 50 * np.sin(t * 0.5))
            y = int(200 + 30 * np.cos(t * 0.3))
            cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
            cv2.putText(frame, "Slow Movement", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif 12 < t < 18:  # Fast movement (anomaly)
            x = int(200 + 100 * np.sin(t * 3))
            y = int(150 + 80 * np.cos(t * 2))
            cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), (0, 0, 255), -1)
            cv2.putText(frame, "FAST MOVEMENT (ANOMALY)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elif 20 < t < 25:  # Multiple objects
            for i in range(3):
                x = int(150 + i*100 + 30 * np.sin(t * 2 + i))
                y = int(200 + 40 * np.cos(t * 1.5 + i))
                cv2.circle(frame, (x, y), 15, (255, 255, 0), -1)
            cv2.putText(frame, "Multiple Objects", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        elif 27 < t < 30:  # Sudden appearance
            x = int(300 + 50 * np.sin(t * 5))
            y = int(100 + 30 * np.cos(t * 4))
            cv2.ellipse(frame, (x, y), (25, 15), 0, 0, 360, (255, 0, 255), -1)
            cv2.putText(frame, "Sudden Appearance", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {t:.1f}s", (width-150, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    return output_path


def test_video_analysis(video_path: str, output_dir: str = "test_output") -> dict:
    """
    Test the CCTV analyzer on a video file.
    
    Args:
        video_path: Path to test video
        output_dir: Output directory for results
        
    Returns:
        Analysis results
    """
    print(f"\n{'='*60}")
    print("TESTING CCTV SUMMARIZATION SYSTEM")
    print(f"{'='*60}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Input Video: {video_path}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Total Frames: {frame_count}")
    
    # Initialize analyzer with different thresholds for testing
    print(f"\nInitializing analyzer...")
    analyzer = CCTVAnalyzer(
        motion_threshold=0.05,  # Lower threshold for more sensitive detection
        anomaly_threshold=1.5   # Lower threshold for more anomaly detection
    )
    
    # Run analysis
    start_time = time.time()
    try:
        results = analyzer.analyze_video(video_path, output_dir)
        analysis_time = time.time() - start_time
        
        print(f"\n{'='*40}")
        print("ANALYSIS RESULTS")
        print(f"{'='*40}")
        print(f"Analysis time: {analysis_time:.1f} seconds")
        print(f"Original duration: {results['total_duration']:.1f} seconds")
        print(f"Compression ratio: {results['total_duration']/len(results['anomaly_timestamps']):.1f}x")
        print(f"Anomalies detected: {results['anomalies_detected']}")
        print(f"Keyframes extracted: {results['keyframes_extracted']}")
        
        if results['anomaly_timestamps']:
            print(f"Anomaly timestamps: {[f'{t:.1f}s' for t in results['anomaly_timestamps']]}")
        
        print(f"\nMotion Statistics:")
        motion_stats = results['motion_statistics']
        print(f"  Mean motion: {motion_stats['mean_motion']:.3f}")
        print(f"  Max motion: {motion_stats['max_motion']:.3f}")
        print(f"  Motion std: {motion_stats['motion_std']:.3f}")
        
        print(f"\nOutput files:")
        print(f"  Summary video: {results['summary_video']}")
        print(f"  Keyframes: {results['keyframes_dir']}")
        print(f"  Report: {output_dir}/analysis_report.json")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


def test_performance_metrics(video_path: str, results: dict) -> dict:
    """
    Calculate performance metrics for the analysis.
    
    Args:
        video_path: Path to original video
        results: Analysis results
        
    Returns:
        Performance metrics
    """
    # Get file sizes
    original_size = os.path.getsize(video_path)
    summary_size = os.path.getsize(results['summary_video']) if os.path.exists(results['summary_video']) else 0
    
    # Calculate metrics
    metrics = {
        'original_size_mb': original_size / (1024 * 1024),
        'summary_size_mb': summary_size / (1024 * 1024),
        'compression_ratio': original_size / max(summary_size, 1),
        'duration_compression': results['total_duration'] / max(len(results['anomaly_timestamps']), 1),
        'anomaly_detection_rate': results['anomalies_detected'] / results['total_duration'],
        'keyframe_efficiency': results['keyframes_extracted'] / results['total_duration']
    }
    
    print(f"\n{'='*40}")
    print("PERFORMANCE METRICS")
    print(f"{'='*40}")
    print(f"Original size: {metrics['original_size_mb']:.1f} MB")
    print(f"Summary size: {metrics['summary_size_mb']:.1f} MB")
    print(f"Size compression: {metrics['compression_ratio']:.1f}x")
    print(f"Duration compression: {metrics['duration_compression']:.1f}x")
    print(f"Anomaly detection rate: {metrics['anomaly_detection_rate']:.2f} events/second")
    print(f"Keyframe efficiency: {metrics['keyframe_efficiency']:.2f} keyframes/second")
    
    return metrics


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test CCTV Summarization System')
    parser.add_argument('--video', '-v', help='Path to test video (optional)')
    parser.add_argument('--create-test', action='store_true', 
                       help='Create synthetic test video')
    parser.add_argument('--output', '-o', default='test_output', 
                       help='Output directory')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration of synthetic test video')
    
    args = parser.parse_args()
    
    try:
        video_path = args.video
        
        # Create test video if requested or no video provided
        if args.create_test or not video_path:
            test_video_path = os.path.join(args.output, "test_video.mp4")
            os.makedirs(args.output, exist_ok=True)
            video_path = create_test_video(test_video_path, args.duration)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return 1
        
        # Run analysis
        results = test_video_analysis(video_path, args.output)
        
        # Calculate performance metrics
        metrics = test_performance_metrics(video_path, results)
        
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Check the output directory: {args.output}")
        print("Files created:")
        print(f"  - Summary video: {results['summary_video']}")
        print(f"  - Keyframes: {results['keyframes_dir']}")
        print(f"  - Analysis report: {args.output}/analysis_report.json")
        
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
