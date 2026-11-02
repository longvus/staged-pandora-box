"""Checks the generic comparison bounds and the shared-nuisance control."""
import sys
from fractions import Fraction as F
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'next_paper'))
from p2_index_robustness import latent_model, shrink
from p2_shared_law_controls import generic_comparison, nuisance_model
from staged_pandora_exact import clear_caches


class GenericComparisonTests(unittest.TestCase):
    def tearDown(self):
        clear_caches()

    def test_generic_bound_covers_quadratic_index_loss(self):
        p,lam=F(4,5),F(1,10)
        m=latent_model(shrink((F(4,5),F(1,5)),p,lam),
                       shrink((F(7,10),F(1,5)),p,lam))
        r=generic_comparison(m)
        self.assertEqual(r['regret'],F(3,125000))
        self.assertEqual(r['root_joint_product_tv'],F(12,125)*lam**2)
        self.assertEqual(r['generic_root_tv_bound'],F(5184,625)*lam**2)

    def test_generic_rate_does_not_need_noisy_signals(self):
        p,lam=F(4,5),F(1,10)
        m=latent_model(shrink((F(4,5),F(1,5)),p,lam),
                       shrink((F(7,10),F(1,5)),p,lam),reliability=(F(1),F(1)))
        self.assertEqual(generic_comparison(m)['root_joint_product_tv'],F(3,3125))

    def test_call_refines_tv_but_not_generic_value_equivalence(self):
        m=nuisance_model()
        r=m.verify()|generic_comparison(m)
        self.assertEqual(r['regret'],0)
        self.assertEqual(r['utility'],F(177,250))
        self.assertEqual(r['root_joint_product_tv'],F(1,2))
        self.assertEqual(r['call_uniform'],0)
        self.assertEqual(r['tv_uniform'],F(28,5))
        self.assertGreater(r['generic_occupancy_bound'],0)


if __name__ == '__main__':
    unittest.main()
