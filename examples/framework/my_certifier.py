# my_certifier.py: your own safety check, shown in the gym next to the built-in ones.
import numpy as np
from lcert.framework import CertifierOutput, NominalInflatedCertifier, register_certifier
from lcert.framework.gym import Scenario, register_scenario


@register_certifier("cushioned_nominal")
class CushionedNominal:
    """The naive model's answer, minus a fixed safety cushion (bigger cushion = more cautious)."""
    name = "cushioned_nominal"

    def __init__(self, cushion=0.3):
        self.cushion = cushion

    def certify(self, env, grid):  # must never look at the true answer
        v = NominalInflatedCertifier(k=0.0).certify(env, grid).value
        return CertifierOutput(np.asarray(v) - self.cushion, {"cushion": self.cushion})


register_scenario(Scenario(
    "my_cart", "My certifier vs the naive one (cart vs wall)", "di_hidden_physics", {"horizon": 2.5},
    [{"label": "naive nominal model (k=0)", "certifier": "nominal_inflated", "params": {"k": 0.0}},
     {"label": "my cushioned certifier (0.3)", "certifier": "cushioned_nominal", "params": {"cushion": 0.3}}],
    render="cart", manual={"x": 0}, goal=[0.15, 0.0], start=[2.0, -0.5], start_box=([0.2, -1.5], [2.8, 1.0]),
    user_defined=True, blurb="Made with examples/framework/my_certifier.py"))
