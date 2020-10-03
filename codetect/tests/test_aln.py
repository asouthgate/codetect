import unittest
from pycodetect.aln import ReadAln

class TestAln(unittest.TestCase):

    def test_get_aln_segments(self):
        a = ReadAln("test")
        bps = [(100,0),(101,1),(102,3),
               (105,3),(106,2),(107,3)]
        for p, b in bps:
            a.append_mapped_base(p, b)
        true_segments = ["ACT","TGT"]
        for si, seg in enumerate(a.get_aln_segments()):
            self.assertEqual(true_segments[si], seg)

    def test_get_aln_tuples(self):
        a = ReadAln("test")
        bps = [(100,0),(101,1),(102,3),
               (105,3),(106,2),(107,3)]
        for p, b in bps:
            a.append_mapped_base(p, b)
        for si, tup in enumerate(a.get_aln_tuples()):
            self.assertEqual(bps[si], tup)      
