"""
Download Allegro Hand MJCF and add a cube + table to the scene.

Usage:
    python scripts/build_allegro_xml.py --out assets/allegro_cube.xml

Requires: mujoco_menagerie (or manual download of allegro_hand.xml)
    pip install mujoco-menagerie   # or clone from GitHub
"""

import argparse
import os
import shutil

MENAGERIE_ALLEGRO = 'mujoco_menagerie/wonik_allegro/left_hand.xml'

CUBE_XML = """
  <!-- Table -->
  <body name="table" pos="0 0 -0.01">
    <geom type="box" size="0.3 0.3 0.01" rgba="0.8 0.7 0.6 1" contype="1" conaffinity="1"/>
  </body>

  <!-- Cube -->
  <body name="cube" pos="0.0 0.05 0.03">
    <freejoint name="cube_free"/>
    <geom type="box" size="0.025 0.025 0.025" rgba="0.9 0.2 0.2 1"
          mass="0.05" contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
  </body>
"""


def build(out_path: str):
    if not os.path.exists(MENAGERIE_ALLEGRO):
        print(f"ERROR: {MENAGERIE_ALLEGRO} not found.")
        print("Clone mujoco_menagerie first:")
        print("  git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        return

    # Output must live next to the source XML so includes + meshes resolve.
    src_dir  = os.path.dirname(os.path.abspath(MENAGERIE_ALLEGRO))
    final    = os.path.join(src_dir, 'allegro_cube.xml')

    with open(MENAGERIE_ALLEGRO, 'r') as f:
        xml = f.read()

    xml = xml.replace('</worldbody>', CUBE_XML + '\n</worldbody>')

    with open(final, 'w') as f:
        f.write(xml)

    import mujoco
    m = mujoco.MjModel.from_xml_path(final)

    # Also write a pointer file at the requested out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(final)   # plain text: absolute path to real XML

    print(f"✓ Built  {final}")
    print(f"  Path written to {out_path}  (nq={m.nq}, nu={m.nu}, nbody={m.nbody})")
    print(f"\nUse this XML path in env.py / bc_train.py / rl_finetune.py:")
    print(f"  {final}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='assets/allegro_cube.xml')
    args = parser.parse_args()
    build(args.out)