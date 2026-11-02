import sys
from fractions import Fraction as F
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'next_paper'))
from p2_index_robustness import Joint, Spec, latent_model, shrink
from staged_pandora_exact import clear_caches, nested


class RobustnessTests(unittest.TestCase):
    def tearDown(self):
        clear_caches()

    def test_independent_reduces_to_exact_theorem(self):
        m = latent_model((F(3,10),)*2, (F(1,2),)*2)
        r = m.verify()
        self.assertEqual((r['regret'],r['max_call'],r['max_tv']), (0,0,0))

    def test_pure_information_sharp_refined_bound(self):
        m = Joint(((((F(h),),(F(h),)), F(1,2)) for h in (0,1)),
                  (Spec((F(3,5),),(True,)),Spec((F(1,100),),(False,))))
        r = m.verify()
        self.assertEqual(r['regret'], F(19,100))
        self.assertEqual(r['refined_bound'], r['regret'])
        self.assertEqual(m.local_action(m.root), -1)

    def test_pairwise_independence_and_zero_root_sharing_do_not_suffice(self):
        worlds = tuple((((F(u),),(F(a),),(F(u^a),)),F(1,4))
                       for u in (0,1) for a in (0,1))
        m = Joint(worlds,(Spec((F(1,100),),(False,)),
                         Spec((F(3,5),),(True,)),Spec((F(3,5),),(True,))))
        r = m.verify()
        self.assertEqual(r['regret'], F(1,25))
        self.assertTrue(all(m.metrics(m.root,i)[1:] == (0,0,0) for i in range(3)))
        self.assertGreater(r['max_call'],0)

    def test_fixed_marginal_quadratic_loss(self):
        for lam in (F(0),F(1,100),F(1,10),F(1,2)):
            p = F(4,5)
            m = latent_model(shrink((F(4,5),F(1,5)),p,lam),
                             shrink((F(7,10),F(1,5)),p,lam))
            self.assertEqual([nested(n)[0] for n in m.nodes(m.root)], [F(1,2)]*2)
            r = m.verify()
            self.assertEqual(r['regret'], F(3,1250)*lam**2)

    def test_negative_correction_occurs(self):
        m = latent_model((F(4,5),F(1,5)), (F(7,10),F(1,5)))
        self.assertLess(m.verify()['min_gamma'],0)
        self.assertEqual(m.trace(m.root,'optimal')[3],0)
        self.assertGreater(m.trace(m.root,'index')[4],0)

    def test_exhaustion_and_single_arm(self):
        m = Joint(((((F(0),F(1)),),F(1)),),
                  (Spec((F(1,10),F(1,10)),(False,True)),))
        r = m.verify()
        self.assertEqual(r['regret'],0)
        self.assertEqual(r['max_call'],0)
        self.assertEqual(r['utility'],F(4,5))


if __name__ == '__main__':
    unittest.main()
