from deepeval.metrics.faithfulness import FaithfulnessTemplate


class PolicyFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_verdicts(*args, **kwargs):
        prompt = FaithfulnessTemplate.generate_verdicts(*args, **kwargs)

        return prompt + """
Additional logical evaluation rules:
- Preserve negation when interpreting each claim.
- Distinguish a rejected request from a permitted policy limit.
- If an action is forbidden beyond a boundary, rejecting it at
  a later point is consistent with that policy.
- A rejection at a later point does not imply permission at
  earlier points or redefine the policy boundary.
- Accept logical implications of the supplied facts.
- Still reject claims that permit a forbidden action or state
  a policy limit that contradicts the supplied facts.
Keep the original output format and verdict schema.
"""
