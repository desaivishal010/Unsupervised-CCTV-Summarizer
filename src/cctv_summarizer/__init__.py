"""
CCTV Summarization System

An efficient CPU-based system for automatically condensing hours of CCTV footage 
into highlight clips by detecting movement and anomalies.
"""

from .core import CCTVAnalyzer

__version__ = "1.0.0"
__author__ = "CCTV Summarization Team"
__email__ = "contact@cctv-summarizer.com"

__all__ = ["CCTVAnalyzer"]