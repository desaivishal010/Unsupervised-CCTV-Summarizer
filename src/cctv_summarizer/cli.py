"""
Command-line interface for CCTV Summarization System.
"""

import argparse
import sys
from .core import CCTVAnalyzer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='CCTV Video Summarization')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--motion-threshold', type=float, default=0.1, 
                       help='Motion detection threshold')
    parser.add_argument('--anomaly-threshold', type=float, default=2.0,
                       help='Anomaly detection threshold')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CCTVAnalyzer(
        motion_threshold=args.motion_threshold,
        anomaly_threshold=args.anomaly_threshold
    )
    
    # Analyze video
    try:
        results = analyzer.analyze_video(args.video_path, args.output)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Original video: {results['total_duration']:.1f} seconds")
        print(f"Anomalies detected: {results['anomalies_detected']}")
        print(f"Keyframes extracted: {results['keyframes_extracted']}")
        print(f"Summary video: {results['summary_video']}")
        print(f"Keyframes saved to: {results['keyframes_dir']}")
        print(f"Analysis report: {args.output}/analysis_report.json")
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
