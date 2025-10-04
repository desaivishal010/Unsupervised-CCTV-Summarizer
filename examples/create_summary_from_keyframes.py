#!/usr/bin/env python3
"""
Script to create summary video directly from extracted keyframe images.
"""

import os
import sys
import argparse
from cctv_summarizer import CCTVAnalyzer


def create_summary_from_keyframes(keyframes_dir: str, output_path: str, 
                                fps: float = 30.0, frame_duration: float = 1.0):
    """
    Create summary video from keyframe images.
    
    Args:
        keyframes_dir: Directory containing keyframe images
        output_path: Path for output summary video
        fps: Frames per second for output video
        frame_duration: Duration to display each keyframe in seconds
    """
    analyzer = CCTVAnalyzer()
    
    print(f"Creating summary video from keyframes in: {keyframes_dir}")
    print(f"Output video: {output_path}")
    print(f"Frame duration: {frame_duration}s per keyframe")
    print(f"FPS: {fps}")
    
    try:
        result_path = analyzer.create_summary_from_keyframes(
            keyframes_dir, output_path, fps, frame_duration
        )
        
        print(f"\n✅ Summary video created successfully: {result_path}")
        
        # Get file size
        file_size = os.path.getsize(result_path) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        
        return result_path
        
    except Exception as e:
        print(f"❌ Error creating summary video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Create summary video from keyframe images')
    parser.add_argument('keyframes_dir', help='Directory containing keyframe images')
    parser.add_argument('--output', '-o', default='keyframe_summary.mp4', 
                       help='Output video path')
    parser.add_argument('--fps', type=float, default=30.0, 
                       help='Frames per second for output video')
    parser.add_argument('--duration', type=float, default=1.0, 
                       help='Duration to display each keyframe in seconds')
    
    args = parser.parse_args()
    
    # Check if keyframes directory exists
    if not os.path.exists(args.keyframes_dir):
        print(f"❌ Error: Keyframes directory not found: {args.keyframes_dir}")
        return 1
    
    # Count keyframe images
    import glob
    keyframe_files = glob.glob(os.path.join(args.keyframes_dir, "keyframe_*.jpg"))
    print(f"Found {len(keyframe_files)} keyframe images")
    
    if len(keyframe_files) == 0:
        print("❌ No keyframe images found!")
        return 1
    
    # Create summary video
    result = create_summary_from_keyframes(
        args.keyframes_dir, 
        args.output, 
        args.fps, 
        args.duration
    )
    
    if result:
        print(f"\n🎬 Summary video ready: {result}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
