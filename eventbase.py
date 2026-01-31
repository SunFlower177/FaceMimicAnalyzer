from metavision_sdk_stream import Camera, CameraStreamSlicer


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.input_event_file:
        camera = Camera.from_file(args.input_event_file)
    else:
        camera = Camera.from_first_available()

    slicer = CameraStreamSlicer(camera.move())
    for _ in slicer:
        print("Events are available!")

if __name__ == "__main__":
    main()