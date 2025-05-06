#!/usr/bin/env python3
"""
Script to download a YouTube video with audio locally using yt-dlp.

Usage:
    python download_youtube_video.py <YouTube_URL> [-o OUTPUT_DIR]

Example:
    python download_youtube_video.py https://www.youtube.com/watch?v=abcdef12345 -o downloads
"""
import argparse
import os
import shutil
from yt_dlp import YoutubeDL

def download_video(url: str, output_path: str = '.') -> None:
    """
    Download a YouTube video with audio.

    :param url: YouTube video URL
    :param output_path: Directory where the video will be saved
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Check if ffmpeg is available for merging formats
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print("Warning: 'ffmpeg' not found. Downloading best available muxed format only.")

    # Choose format: merge video+audio if ffmpeg is present, else download best combined
    format_option = 'bestvideo+bestaudio/best' if has_ffmpeg else 'best'

    # Prepare options
    ydl_opts = {
        'format': format_option,
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'progress_hooks': [progress_hook],
    }
    # Only set merge_output_format if ffmpeg is available
    if has_ffmpeg:
        ydl_opts['merge_output_format'] = 'mp4'

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def progress_hook(d):
    """
    Hook to report download progress.
    """
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', '').strip()
        speed = d.get('_speed_str', '').strip()
        eta = d.get('_eta_str', '').strip()
        print(f"Downloading: {percent} at {speed}, ETA: {eta}", end='\r')
    elif d['status'] == 'finished':
        if 'merge_output_format' in d.get('info_dict', {}):
            print(f"\nDownload completed, now merging formats...")
        else:
            print(f"\nDownload completed.")


def main():
    parser = argparse.ArgumentParser(
        description='Download a YouTube video with audio locally'
    )
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument(
        '-o', '--output', default='.',
        help='Output directory (default: current directory)'
    )
    args = parser.parse_args()

    download_video(args.url, args.output)


if __name__ == '__main__':
    main()
