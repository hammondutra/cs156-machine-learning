import unittest
import pandas as pd
import numpy as np
from src.features import validate_matches, normalize_names, vocabulary, encode

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame([{"Match_ID":1,"Blue_1":"COLT","Blue_2":"SPIKE","Blue_3":"LILY",
        "Red_1":"COLT","Red_2":"CLANCY","Red_3":"DYNAMIKE","Blue_Win":1,"Result":"Victory"}])
    def test_validation(self):
        self.assertTrue(validate_matches(self.df))
    def test_signed_cancel(self):
        vocab=vocabulary(self.df)
        x=encode(self.df,vocab)
        self.assertEqual(x[0,vocab.index("COLT")],0)
        self.assertEqual(x[0,vocab.index("SPIKE")],1)
        self.assertEqual(x[0,vocab.index("DYNAMIKE")],-1)
    def test_slot_order(self):
        changed=self.df.copy()
        changed["Blue_1"],changed["Blue_3"]=self.df["Blue_3"],self.df["Blue_1"]
        vocab=vocabulary(self.df)
        np.testing.assert_array_equal(encode(self.df,vocab),encode(changed,vocab))
    def test_side_swap(self):
        changed=self.df.copy()
        for j in range(1,4):
            changed[f"Blue_{j}"]=self.df[f"Red_{j}"]
            changed[f"Red_{j}"]=self.df[f"Blue_{j}"]
        vocab=vocabulary(self.df)
        np.testing.assert_array_equal(encode(self.df,vocab),-encode(changed,vocab))

if __name__=="__main__":
    unittest.main()
