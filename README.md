# CCTV Summarization System

An efficient CPU-based system for automatically condensing hours of CCTV footage into highlight clips by detecting movement and anomalies.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Motion Detection**: Uses optical flow (Farneback) for efficient motion tracking
- **Anomaly Detection**: Statistical analysis of motion patterns and frame differences
- **Keyframe Extraction**: Intelligent sampling based on activity levels
- **Video Segmentation**: Groups related events and removes redundancy
- **CPU Optimized**: Designed to run efficiently on CPU without GPU requirements
- **Multi-Resolution Support**: Works with videos from 320x240 to 4K+ resolution

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cctv-summarizer.git
cd cctv-summarizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from cctv_summarizer import CCTVAnalyzer

# Initialize analyzer
analyzer = CCTVAnalyzer(
    motion_threshold=0.1,    # Motion detection sensitivity
    anomaly_threshold=2.0     # Anomaly detection sensitivity
)

# Analyze video
results = analyzer.analyze_video("path/to/video.mp4", "output/")

print(f"Anomalies detected: {results['anomalies_detected']}")
print(f"Keyframes extracted: {results['keyframes_extracted']}")
```

### Command Line Usage

```bash
# Basic analysis
cctv-analyze path/to/video.mp4 --output results/

# Custom thresholds
cctv-analyze video.mp4 --motion-threshold 0.05 --anomaly-threshold 1.5
```

## How It Works

### 1. Motion Feature Extraction
- Processes video frames using optical flow
- Calculates motion magnitude and frame differences
- Skips frames for efficiency (processes every 5th frame by default)

### 2. Anomaly Detection
- Uses statistical analysis (z-score) on motion features
- Identifies events that deviate significantly from normal patterns
- Configurable threshold for sensitivity

### 3. Keyframe Extraction
- Combines anomaly timestamps with high-motion events
- Removes duplicate timestamps
- Limits to maximum number of keyframes for efficiency

### 4. Summary Video Creation
- Creates condensed video with timestamp overlays
- Saves individual keyframe images
- Generates detailed analysis reports

## Examples

### Example 1: Basic Analysis

```python
from cctv_summarizer import CCTVAnalyzer

analyzer = CCTVAnalyzer()
results = analyzer.analyze_video("surveillance.mp4", "output/")
```

### Example 2: Custom Configuration

```python
analyzer = CCTVAnalyzer(
    motion_threshold=0.05,    # More sensitive motion detection
    anomaly_threshold=1.5     # More sensitive anomaly detection
)

results = analyzer.analyze_video("video.mp4", "results/")
```

### Example 3: Batch Processing

```python
import os
from cctv_summarizer import CCTVAnalyzer

analyzer = CCTVAnalyzer()

video_dir = "videos/"
output_dir = "results/"

for video_file in os.listdir(video_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, video_file.split('.')[0])
        
        results = analyzer.analyze_video(video_path, output_path)
        print(f"Processed {video_file}: {results['anomalies_detected']} anomalies")
```

## Configuration

### Motion Detection Threshold
- `motion_threshold`: Sensitivity for motion detection (0.0-1.0, default: 0.1)
- Lower values = more sensitive to small movements

### Anomaly Detection Threshold
- `anomaly_threshold`: Z-score threshold for anomalies (default: 2.0)
- Lower values = more sensitive to anomalies

## Output Files

The system generates several output files:

- `summary.mp4`: Condensed video with key segments
- `keyframes/`: Directory with individual keyframe images
- `analysis_report.json`: Detailed analysis results and statistics

## Performance

### Expected Performance:
- **Processing Speed**: ~2-5x real-time on modern CPU
- **Memory Usage**: Low memory footprint
- **Compression Ratio**: 10-50x reduction in video length
- **Accuracy**: High precision for motion and anomaly detection

### Supported Resolutions:
- 320x240 (standard surveillance)
- 640x480 (enhanced surveillance)
- 1920x1080 (HD)
- 2160x3840 (4K+)

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.21+
- SciPy 1.7+
- scikit-learn 1.0+
- matplotlib 3.5+

## Development

### Running Tests

```bash
# Run example scripts
python examples/test_cctv_summarizer.py --create-test

# Test with your own video
python examples/test_cctv_summarizer.py --video path/to/video.mp4
```

### Project Structure

```
cctv-summarizer/
├── src/
│   └── cctv_summarizer/
│       ├── __init__.py
│       ├── core.py          # Main analysis functionality
│       └── cli.py           # Command-line interface
├── examples/
│   ├── test_cctv_summarizer.py
│   └── create_summary_from_keyframes.py
├── tests/
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for excellent computer vision tools
- SciPy and scikit-learn for statistical analysis capabilities
- The computer vision research community for optical flow algorithms