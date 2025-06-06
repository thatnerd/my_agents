#!/usr/bin/env python

import unittest
import sys
import os
sys.path.append(os.getcwd())
import script.module_name as mn


class TestModuleName(unittest.TestCase):

    def test_func(self):
        pass


if __name__ == '__main__':
    unittest.main()
