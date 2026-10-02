"""Run with:  python -m pytest -q   (or: python -m unittest discover -s tests)"""
from __future__ import annotations

import asyncio
import time
import unittest

from vox.clock import now_ms, run_virtual
from vox.config import PRESETS
from vox.deciders.base import TurnAnalysis
from vox.deciders.fallback import FallbackDecider
from vox.deciders.heuristic import HeuristicDecider, eot_probability, overlap_kind, route_tool
from vox.deciders.jev import JevDecider, build_turn_request, parse_turn_response
from vox.phi import Redactor
from vox.sim.runner import run_one
from vox.sim.scenarios import SCENARIOS, parse_turn


class TestClock(unittest.TestCase):
    def test_virtual_time_is_instant_and_exact(self):
        async def main():
            t0 = now_ms()
            await asyncio.sleep(3600)
            return now_ms() - t0
        wall = time.time()
        elapsed = run_virtual(main())
        self.assertAlmostEqual(elapsed, 3_600_000, delta=1)
        self.assertLess(time.time() - wall, 1.0)


class TestPHI(unittest.TestCase):
    def test_redacts_identifiers(self):
        r = Redactor(["Priya Raman"])
        out = r.redact("hi this is Priya Raman, DOB 03/14/1985, call me at 415-555-0199, MRN 88231945")
        for leak in ("Priya", "Raman", "03/14/1985", "555-0199", "88231945"):
            self.assertNotIn(leak, out)
        self.assertIn("[NAME]", out)
        self.assertIn("[PHONE]", out)

    def test_pseudonym_stable_and_opaque(self):
        r = Redactor(salt="s")
        self.assertEqual(r.pseudonym("Priya"), r.pseudonym("priya"))
        self.assertNotIn("priya", r.pseudonym("Priya").lower())


class TestHeuristic(unittest.TestCase):
    def test_eot(self):
        self.assertLess(eot_probability("I wanted to ask about my"), 0.2)
        self.assertLess(eot_probability("so um"), 0.2)
        self.assertGreater(eot_probability("can you repeat that"), 0.8)
        self.assertGreater(eot_probability("okay thank you"), 0.8)

    def test_routing_and_overlap(self):
        self.assertEqual(route_tool("I feel dizzy since the new pills")[0], "report_side_effect")
        self.assertEqual(route_tool("can we move it to friday")[0], "reschedule_appointment")
        self.assertEqual(overlap_kind("mm-hmm")[0], "backchannel")
        self.assertEqual(overlap_kind("wait sorry")[0], "interruption")

    def test_parse_turn(self):
        self.assertEqual(parse_turn("a <600> b c"), [("a", 600.0), ("b", None), ("c", None)])


class TestJev(unittest.TestCase):
    RESP = {"model": "jev-1.13.0", "usage": {"input_tokens": 90, "output_tokens": 0}, "answers": {
        "eot": {"type": "noul", "noul": 0.91},
        "tool": {"type": "choice", "choice": "medication_info", "probabilities": {}, "confidence": 0.8},
        "urgency": {"type": "score", "score": 1.0, "legend": {"0": "a", "1": "b", "2": "c"},
                    "probabilities": {}, "confidence": 0.7}}}

    def test_request_shape(self):
        body = build_turn_request("my metformin refill", "", "jev-latest")
        self.assertEqual(set(body["questions"]), {"eot", "tool", "urgency"})
        self.assertEqual(body["questions"]["eot"]["type"], "noul")
        self.assertIn("medication_info", body["questions"]["tool"]["criteria"])
        self.assertEqual(len(body["questions"]["urgency"]["criteria"]), 3)

    def test_parse(self):
        self.assertEqual(parse_turn_response(self.RESP), (0.91, "medication_info", 0.8, 1.0))

    def test_client_redacts_phi_before_sending(self):
        sent = {}

        async def fake_post(url, body, headers, timeout):
            sent.update(body=body, headers=headers)
            return self.RESP

        async def main():
            d = JevDecider(api_key="k", redactor=Redactor(["Priya Raman"]), post=fake_post)
            return await d.analyze_turn("this is Priya Raman, my refill please")
        a = asyncio.run(main())
        self.assertEqual(a.tool, "medication_info")
        self.assertNotIn("Priya", str(sent["body"]))
        self.assertEqual(sent["headers"]["Authorization"], "Bearer k")

    def test_fallback_on_slow_primary(self):
        class Slow(HeuristicDecider):
            name = "slow"

            async def analyze_turn(self, transcript, agent_last=""):
                await asyncio.sleep(5)
                return TurnAnalysis(0.0)

        async def main():
            d = FallbackDecider(Slow(), HeuristicDecider(), budget_ms=300)
            t0 = now_ms()
            a = await d.analyze_turn("okay thank you")
            return a, now_ms() - t0, d
        a, took, d = run_virtual(main())
        self.assertTrue(a.fallback)
        self.assertLess(took, 320)
        self.assertGreater(a.eot_prob, 0.8)
        self.assertEqual(d.stats.fallbacks, 1)


class TestOrchestrator(unittest.TestCase):
    def test_baseline_waits_for_timeout(self):
        res = run_one(SCENARIOS[0], 0, PRESETS["baseline-700"], "heuristic")
        lats = [r["latency_ms"] for r in res.rows if r.get("latency_ms") is not None]
        self.assertEqual(len(lats), len(SCENARIOS[0].turns))
        self.assertTrue(all(lat >= 700 for lat in lats))

    def test_hybrid_is_faster_and_ignores_backchannels(self):
        base, hyb = [], []
        bc_stops = 0
        for sc in SCENARIOS:
            for r in run_one(sc, 0, PRESETS["baseline-700"], "heuristic").rows:
                if r.get("latency_ms") is not None:
                    base.append(r["latency_ms"])
            res = run_one(sc, 0, PRESETS["hybrid"], "heuristic")
            hyb += [r["latency_ms"] for r in res.rows if r.get("latency_ms") is not None]
            bc_stops += sum(o["stopped"] for o in res.ov_rows if o["type"] == "backchannel")
        self.assertLess(sorted(hyb)[len(hyb) // 2], sorted(base)[len(base) // 2] - 500)
        self.assertEqual(bc_stops, 0)

    def test_every_response_is_resolved(self):
        res = run_one(SCENARIOS[3], 1, PRESETS["hybrid"], "heuristic")
        for r in res.orch.responses:
            self.assertNotEqual(r.outcome, "pending", f"response {r.id} never resolved")

    def test_traces_contain_no_patient_name(self):
        sc = SCENARIOS[0]
        res = run_one(sc, 0, PRESETS["hybrid"], "heuristic")
        dump = str([s.attrs for s in res.tracer.spans])
        self.assertNotIn("Priya", dump)
        self.assertIn("[NAME]", dump)


if __name__ == "__main__":
    unittest.main()


class TestMiniJev(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vox.minijev.model import MiniJev
        from vox.minijev.train import build_dataset
        train, _ = build_dataset(n_utts=1500)
        cls.m = MiniJev().fit(train)

    def test_answers_have_jev_shapes(self):
        n = self.m.noul_eot("I wanted to ask about my")
        self.assertEqual(n["type"], "noul")
        self.assertTrue(0 <= n["noul"] <= 1)
        c = self.m.choice_tool("can I get a refill on my metformin")
        self.assertEqual(c["type"], "choice")
        self.assertAlmostEqual(sum(c["probabilities"].values()), 1.0, places=5)
        s = self.m.score_urgency("I have chest pain and trouble breathing")
        self.assertEqual(s["type"], "score")
        self.assertIn("legend", s)

    def test_sensible_decisions(self):
        self.assertLess(self.m.noul_eot("can I get a refill on my")["noul"], 0.5)
        self.assertEqual(self.m.choice_tool("I need to move my appointment to thursday")["choice"],
                         "reschedule_appointment")
        self.assertGreater(self.m.score_urgency("I have really bad chest pain")["score"], 1.0)
        self.assertEqual(self.m.choice_overlap("mm-hmm")["choice"], "backchannel")
        self.assertEqual(self.m.choice_overlap("wait sorry")["choice"], "interruption")

    def test_no_benchmark_sentences_in_training(self):
        from vox.minijev.data import _excluded, utterances
        import random
        gen = {u[0] for u in utterances(3000, random.Random(1))}
        self.assertFalse(gen & _excluded())
