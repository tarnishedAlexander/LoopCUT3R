#!/usr/bin/env python3
"""Extract RGB frames from a ROS1 bag file for use with demo.py."""

import os
import sys
import argparse
import numpy as np
import cv2
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RGB images from a ROS1 bag file.")
    parser.add_argument("bag", help="Path to .bag file")
    parser.add_argument("out_dir", help="Output directory for extracted frames")
    parser.add_argument("--topic", default="/camera/color/image_raw",
                        help="Image topic to extract (default: /camera/color/image_raw)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Extract every Nth frame (default: 5)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to extract")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    saved = 0
    seen = 0

    with Reader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            print(f"ERROR: topic '{args.topic}' not found in bag.")
            print("Available image topics:")
            for c in reader.connections:
                if "Image" in c.msgtype:
                    print(f"  {c.topic}  [{c.msgtype}]")
            sys.exit(1)

        total = sum(1 for _ in reader.messages(connections=conns))
        print(f"Found {total} frames on {args.topic}, extracting every {args.stride}th frame")
        expected = (total + args.stride - 1) // args.stride
        if args.max_frames:
            expected = min(expected, args.max_frames)
        print(f"Expecting ~{expected} output frames")

        for conn, ts, data in reader.messages(connections=conns):
            if seen % args.stride == 0:
                msg = typestore.deserialize_ros1(data, conn.msgtype)

                # Convert sensor_msgs/Image to numpy array
                h, w = msg.height, msg.width
                encoding = msg.encoding

                raw = np.frombuffer(msg.data, dtype=np.uint8)

                if encoding in ("rgb8", "bgr8"):
                    img = raw.reshape(h, w, 3)
                    if encoding == "rgb8":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == "rgba8":
                    img = raw.reshape(h, w, 4)
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif encoding in ("mono8", "8UC1"):
                    img = raw.reshape(h, w)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"Unsupported encoding: {encoding}, skipping frame {seen}")
                    seen += 1
                    continue

                fname = os.path.join(args.out_dir, f"frame_{saved:06d}.jpg")
                cv2.imwrite(fname, img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                saved += 1

                if saved % 50 == 0:
                    print(f"  Saved {saved} frames...")

                if args.max_frames and saved >= args.max_frames:
                    break

            seen += 1

    print(f"\nDone. Extracted {saved} frames → {args.out_dir}")


if __name__ == "__main__":
    main()
