# Examples

This directory contains example scripts demonstrating how to use the CCTV Summarization System.

## Files

- `test_cctv_summarizer.py` - Comprehensive test script with synthetic video generation
- `create_summary_from_keyframes.py` - Script to create summary videos from keyframe images

## Usage

### Test with Synthetic Video

```bash
python examples/test_cctv_summarizer.py --create-test --duration 30
```

### Test with Your Own Video

```bash
python examples/test_cctv_summarizer.py --video path/to/your/video.mp4
```

### Create Summary from Keyframes

```bash
python examples/create_summary_from_keyframes.py path/to/keyframes --output summary.mp4
```

## Features

- **Synthetic Video Generation**: Creates test videos with various motion patterns
- **Performance Metrics**: Processing time, compression ratios, efficiency stats
- **Comprehensive Analysis**: Motion statistics, anomaly detection rates
- **Output Validation**: Verifies all generated files
