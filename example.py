#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import sys
from utensor_cgen.snippets import HelloWorld
from utensor_cgen.composer import Composer


def main(output_fname):
  """Main function
  """
  comp = Composer()
  hello_world = HelloWorld()
  comp.add_snippet(hello_world)
  with open(output_fname, "w") as wf:
    wf.write(comp.compose())
  return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--output", dest="output_fname",
                      default="main.cpp", help="output file name (default: %(default)s)", 
                      metavar="FILE.cpp")
  ARGS = vars(parser.parse_args())
  sys.exit(main(**ARGS))
