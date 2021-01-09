# Layers21

* CPU/GPU: Layers21 should have a CPU/GPU agnostic API with generic array variables and no
  CUDNN specific types/values. However internally they may have GPU specific fields/code to
  optimize performance.

* Layers21/Ops21: Layers21 should only depend on basic Julia and the primitive operators in
  Ops21. In particular there should be no direct CUDNN calls. Ops21 should have the
  functional versions of all Layers21 layers (all options in keyword arguments, no state),
  whereas Layer21 should keep state values, weights, and performance optimizing buffers in
  struct fields. This is akin to the torch.nn / torch.nn.functional distinction.
