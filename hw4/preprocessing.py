import os
import argparse


def generate_file_list(degraded_dir, prefix, output_txt, base_dir):
    """Scan degraded_dir for images with given prefix, write relative paths to output_txt."""
    filenames = sorted([
        f for f in os.listdir(degraded_dir)
        if f.startswith(prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    rel_dir = os.path.relpath(degraded_dir, base_dir)
    with open(output_txt, 'w') as f:
        for name in filenames:
            f.write(os.path.join(rel_dir, name) + '\n')

    print(f"[{prefix.rstrip('-')}] {len(filenames)} images -> {output_txt}")
    return len(filenames)


def main(args):
    degraded_dir = os.path.join(args.data_dir, 'train', 'degraded')
    base_dir = args.data_dir

    rain_txt = os.path.join(os.path.dirname(args.data_dir), 'rain.txt')
    snow_txt = os.path.join(os.path.dirname(args.data_dir), 'snow.txt')

    n_rain = generate_file_list(degraded_dir, 'rain-', rain_txt, base_dir)
    n_snow = generate_file_list(degraded_dir, 'snow-', snow_txt, base_dir)

    print(f"\nDone. Total: {n_rain} rain + {n_snow} snow = {n_rain + n_snow} images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/hw4_realse_dataset',
                        help='Root dataset directory (contains train/degraded)')
    args = parser.parse_args()
    main(args)
