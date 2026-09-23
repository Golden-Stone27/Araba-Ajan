# Embedded com.unity.ai.inference 2.6.1 (M2 patch)

Unmodified copy of the registry package except for ONE file:
`Editor/ONNX/Google.Protobuf_Packed.dll.meta` — the editor copy is excluded from all Standalone
platforms (Any + Standalone disabled, Exclude* = 1).

Why: com.unity.ml-agents 4.x ships a runtime `Google.Protobuf_Packed.dll` with the same file name.
In player builds Unity resolved the reference to this editor-only copy and every ML-Agents gRPC file
failed with CS0400 (ml-agents issue #6310; the 4.1.0 fix did not help on Unity 6000.4.6f1).
Documentation~ and Samples~ were dropped (not needed). Remove this folder once upstream fixes it.
