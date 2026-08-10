"""guard_distill — distill ShieldGemma-2B into a fast single-pass K-12 safety guard.

Pipeline: build_corpus -> label_teacher -> train_student (ORCD GPU) ->
eval_student -> serve_student (:8005). See README.md and ORCD_TRAINING_PLAN.md.
"""
