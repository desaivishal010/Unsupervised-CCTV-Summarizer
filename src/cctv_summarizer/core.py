"""
Core CCTV analysis functionality.
"""

import cv2
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from typing import List, Tuple, Dict
import json


class CCTVAnalyzer:
    """
    Efficient CPU-based CCTV video analyzer for unsupervised summarization.
    Focuses on motion detection, anomaly detection, and intelligent keyframe extraction.
    """
    
    def __init__(self, motion_threshold: float = 0.1, anomaly_threshold: float = 2.0):
        """
        Initialize the CCTV analyzer.
        
        Args:
            motion_threshold: Threshold for motion detection (0.0-1.0)
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.motion_threshold = motion_threshold
        self.anomaly_threshold = anomaly_threshold
        
    def extract_motion_features(self, video_path: str) -> Dict:
        """
        Extract motion features from video using optical flow.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing motion data and timestamps
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Processing video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Initialize variables
        motion_scores = []
        timestamps = []
        frame_diffs = []
        prev_gray = None
        
        # Process every 5th frame for efficiency (adjustable)
        frame_skip = 5
        
        with tqdm(total=total_frames//frame_skip, desc="Extracting motion features") as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_gray is not None:
                        # Calculate frame difference
                        frame_diff = cv2.absdiff(gray, prev_gray)
                        diff_score = np.mean(frame_diff) / 255.0
                        frame_diffs.append(diff_score)
                        
                        # Calculate optical flow using Farneback method
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        
                        # Calculate motion magnitude from flow
                        motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        avg_motion = np.mean(motion_magnitude)
                        motion_scores.append(avg_motion)
                        
                        timestamps.append(frame_count / fps)
                    
                    prev_gray = gray.copy()
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        return {
            'motion_scores': np.array(motion_scores),
            'frame_diffs': np.array(frame_diffs),
            'timestamps': np.array(timestamps),
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }
    
    def detect_anomalies(self, motion_data: Dict) -> List[Tuple[float, float]]:
        """
        Detect anomalous events using statistical analysis.
        
        Args:
            motion_data: Motion features from extract_motion_features
            
        Returns:
            List of (timestamp, anomaly_score) tuples
        """
        motion_scores = motion_data['motion_scores']
        frame_diffs = motion_data['frame_diffs']
        timestamps = motion_data['timestamps']
        
        # Combine motion and frame difference features
        features = np.column_stack([motion_scores, frame_diffs])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Calculate anomaly scores using z-score
        anomaly_scores = np.abs(stats.zscore(features_scaled, axis=0))
        combined_anomaly_score = np.mean(anomaly_scores, axis=1)
        
        # Find anomalies above threshold
        anomaly_indices = np.where(combined_anomaly_score > self.anomaly_threshold)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append((timestamps[idx], combined_anomaly_score[idx]))
        
        return anomalies
    
    def extract_keyframes(self, video_path: str, motion_data: Dict, 
                         anomalies: List[Tuple[float, float]], 
                         max_keyframes: int = 20) -> List[Tuple[float, np.ndarray]]:
        """
        Extract keyframes based on motion and anomaly data.
        
        Args:
            video_path: Path to input video
            motion_data: Motion features
            anomalies: List of detected anomalies
            max_keyframes: Maximum number of keyframes to extract
            
        Returns:
            List of (timestamp, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        fps = motion_data['fps']
        
        # Create candidate timestamps
        candidate_timestamps = []
        
        # Add anomaly timestamps
        for timestamp, _ in anomalies:
            candidate_timestamps.append((timestamp, 'anomaly'))
        
        # Add high-motion timestamps
        motion_scores = motion_data['motion_scores']
        timestamps = motion_data['timestamps']
        motion_threshold = np.percentile(motion_scores, 80)  # Top 20% motion
        
        for i, score in enumerate(motion_scores):
            if score > motion_threshold:
                candidate_timestamps.append((timestamps[i], 'motion'))
        
        # Sort by timestamp and remove duplicates
        candidate_timestamps.sort(key=lambda x: x[0])
        unique_timestamps = []
        seen_times = set()
        
        for timestamp, _ in candidate_timestamps:
            if timestamp not in seen_times:
                unique_timestamps.append(timestamp)
                seen_times.add(timestamp)
        
        # Limit to max_keyframes
        if len(unique_timestamps) > max_keyframes:
            # Select most significant timestamps
            step = len(unique_timestamps) // max_keyframes
            unique_timestamps = unique_timestamps[::step][:max_keyframes]
        
        # Extract frames
        keyframes = []
        for timestamp in unique_timestamps:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                keyframes.append((timestamp, frame))
        
        cap.release()
        return keyframes
    
    def create_summary_video(self, video_path: str, keyframes: List[Tuple[float, np.ndarray]], 
                           output_path: str, frame_duration: float = 1.0) -> str:
        """
        Create summary video directly from keyframe images.
        
        Args:
            video_path: Path to original video (for getting video properties)
            keyframes: List of keyframes with timestamps
            output_path: Path for output summary video
            frame_duration: Duration to display each keyframe in seconds
            
        Returns:
            Path to created summary video
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating summary video from {len(keyframes)} keyframes...")
        
        for i, (timestamp, frame) in enumerate(tqdm(keyframes, desc="Creating summary")):
            # Resize frame if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Add timestamp overlay
            cv2.putText(frame, f"Time: {timestamp:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Keyframe {i+1}/{len(keyframes)}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame multiple times for desired duration
            frames_to_write = int(fps * frame_duration)
            for _ in range(frames_to_write):
                out.write(frame)
        
        out.release()
        
        return output_path
    
    def create_summary_from_keyframes(self, keyframes_dir: str, output_path: str, 
                                    fps: float = 30.0, frame_duration: float = 1.0) -> str:
        """
        Create summary video directly from saved keyframe images.
        
        Args:
            keyframes_dir: Directory containing keyframe images
            output_path: Path for output summary video
            fps: Frames per second for output video
            frame_duration: Duration to display each keyframe in seconds
            
        Returns:
            Path to created summary video
        """
        import glob
        import os
        
        # Get all keyframe images
        keyframe_files = sorted(glob.glob(os.path.join(keyframes_dir, "keyframe_*.jpg")))
        
        if not keyframe_files:
            raise ValueError(f"No keyframe images found in {keyframes_dir}")
        
        # Get dimensions from first image
        first_frame = cv2.imread(keyframe_files[0])
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating summary video from {len(keyframe_files)} keyframe images...")
        
        for i, keyframe_file in enumerate(tqdm(keyframe_files, desc="Creating summary")):
            frame = cv2.imread(keyframe_file)
            
            # Extract timestamp from filename
            filename = os.path.basename(keyframe_file)
            timestamp_str = filename.split('_')[2].replace('s.jpg', '')
            timestamp = float(timestamp_str)
            
            # Add timestamp overlay
            cv2.putText(frame, f"Time: {timestamp:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Keyframe {i+1}/{len(keyframe_files)}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame multiple times for desired duration
            frames_to_write = int(fps * frame_duration)
            for _ in range(frames_to_write):
                out.write(frame)
        
        out.release()
        
        return output_path
    
    def analyze_video(self, video_path: str, output_dir: str = "output") -> Dict:
        """
        Complete video analysis pipeline.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            
        Returns:
            Analysis results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Step 1: Extracting motion features...")
        motion_data = self.extract_motion_features(video_path)
        
        print("Step 2: Detecting anomalies...")
        anomalies = self.detect_anomalies(motion_data)
        print(f"Found {len(anomalies)} anomalous events")
        
        print("Step 3: Extracting keyframes...")
        keyframes = self.extract_keyframes(video_path, motion_data, anomalies)
        print(f"Extracted {len(keyframes)} keyframes")
        
        print("Step 4: Creating summary video...")
        summary_path = os.path.join(output_dir, "summary.mp4")
        self.create_summary_video(video_path, keyframes, summary_path)
        
        # Save keyframes as images
        keyframe_dir = os.path.join(output_dir, "keyframes")
        os.makedirs(keyframe_dir, exist_ok=True)
        
        for i, (timestamp, frame) in enumerate(keyframes):
            cv2.imwrite(os.path.join(keyframe_dir, f"keyframe_{i:03d}_{timestamp:.1f}s.jpg"), frame)
        
        # Create analysis report
        report = {
            'video_path': video_path,
            'total_duration': motion_data['duration'],
            'total_frames': motion_data['total_frames'],
            'anomalies_detected': len(anomalies),
            'keyframes_extracted': len(keyframes),
            'summary_video': summary_path,
            'keyframes_dir': keyframe_dir,
            'anomaly_timestamps': [t for t, _ in anomalies],
            'motion_statistics': {
                'mean_motion': float(np.mean(motion_data['motion_scores'])),
                'max_motion': float(np.max(motion_data['motion_scores'])),
                'motion_std': float(np.std(motion_data['motion_scores']))
            }
        }
        
        # Save report
        with open(os.path.join(output_dir, "analysis_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
