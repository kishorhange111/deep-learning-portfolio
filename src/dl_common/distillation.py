"""
Knowledge-distillation loss (Hinton et al., 2015), used by
`nlp/bert-distillation-onnx-quantization`.

    loss = alpha * CE(student_logits, labels)
         + (1 - alpha) * T^2 * KL( softmax(teacher / T) || softmax(student / T) )

* The temperature T > 1 softens both distributions, so the student also learns
  the teacher's *relative* confidence in the wrong classes ("dark knowledge").
* Gradients of the soft term scale like 1 / T^2, so it is multiplied by T^2 to
  keep both terms on a comparable scale when T changes.
"""

import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """Mix of the hard-label cross-entropy and the softened teacher-student KL divergence.

    student_logits, teacher_logits : (batch, classes)
    labels                         : (batch,) integer class ids
    """
    hard = F.cross_entropy(student_logits, labels)
    # kl_div expects log-probabilities for the input and probabilities for the target;
    # "batchmean" divides by the batch size, which matches the mathematical definition of KL.
    soft = (
        F.kl_div(
            F.log_softmax(student_logits / temperature, -1),
            F.softmax(teacher_logits / temperature, -1),
            reduction="batchmean",
        )
        * temperature**2
    )  # T^2 keeps gradient scale comparable
    return alpha * hard + (1 - alpha) * soft
